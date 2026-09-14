#define NOMINMAX

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif

#include <omp.h>


struct Config {
    const char* id;
    int bc;
    int bf;
    double zmin;
    double zmax;
    std::array<int, 32> mapping;
};

#include "p3a_configs.hpp"


static constexpr std::size_t GROUP_SIZE = 32;
static constexpr std::size_t PACKED_BYTES_PER_GROUP = 16;

static constexpr std::size_t GROUPS = (1ull << 22);

static constexpr std::array<int, 3> THREAD_COUNTS = {
    1, 4, 8
};

static constexpr int WARMUP_RUNS = 2;
static constexpr int PAIRED_BLOCKS = 15;

static volatile double GLOBAL_SINK = 0.0;


static inline uint32_t popcount32(uint32_t x) {
#ifdef _MSC_VER
    return static_cast<uint32_t>(__popcnt(x));
#else
    return static_cast<uint32_t>(__builtin_popcount(x));
#endif
}


static inline int decode_q(
    const uint8_t* packed_group,
    int lane
) {
    const uint8_t byte_value =
        packed_group[static_cast<std::size_t>(lane >> 1)];

    const int nibble =
        (lane & 1)
        ? static_cast<int>((byte_value >> 4) & 0x0F)
        : static_cast<int>(byte_value & 0x0F);

    return nibble >= 8
        ? nibble - 16
        : nibble;
}


static uint32_t encode_sbl32(
    double z,
    const Config& cfg
) {
    const int bc = cfg.bc;
    const int bf = cfg.bf;

    if (bc <= 0 || bc >= 32) {
        throw std::runtime_error("invalid Bc");
    }

    if (bf <= 0 || bf >= 32) {
        throw std::runtime_error("invalid Bf");
    }

    if (bc + bf != 32) {
        throw std::runtime_error("Bc+Bf != 32");
    }

    const uint32_t fmax =
        (1u << bf) - 1u;

    const double width =
        cfg.zmax - cfg.zmin;

    if (!(width > 0.0)) {
        throw std::runtime_error("invalid z range");
    }

    const double clipped =
        (std::min)(
            cfg.zmax,
            (std::max)(
                cfg.zmin,
                z
            )
        );

    double u =
        static_cast<double>(bc + 1)
        *
        (clipped - cfg.zmin)
        /
        width;

    const bool endpoint =
        u >= static_cast<double>(bc + 1);

    int coarse =
        static_cast<int>(std::floor(u));

    coarse =
        (std::max)(
            0,
            (std::min)(
                bc,
                coarse
            )
        );

    const double frac =
        u - static_cast<double>(coarse);

    long long fine_ll =
        std::llrint(
            frac
            *
            static_cast<double>(fmax)
        );

    fine_ll =
        std::max<long long>(
            0,
            std::min<long long>(
                static_cast<long long>(fmax),
                fine_ll
            )
        );

    uint32_t fine =
        static_cast<uint32_t>(fine_ll);

    if (endpoint) {
        coarse = bc;
        fine = fmax;
    }

    const uint32_t thermo =
        coarse == 0
        ? 0u
        : ((1u << coarse) - 1u);

    return
        thermo
        |
        (fine << bc);
}


static uint32_t logical_to_physical(
    uint32_t logical_word,
    const Config& cfg
) {
    uint32_t physical_word = 0u;

    for (
        int logical_bit = 0;
        logical_bit < 32;
        ++logical_bit
    ) {
        const int physical_bit =
            cfg.mapping[
                static_cast<std::size_t>(logical_bit)
            ];

        if (
            physical_bit < 0
            ||
            physical_bit >= 32
        ) {
            throw std::runtime_error(
                "physical bit outside 0..31"
            );
        }

        const uint32_t bit =
            (logical_word >> logical_bit) & 1u;

        physical_word |=
            bit << physical_bit;
    }

    return physical_word;
}


struct DecoderData {
    uint32_t identity_coarse_mask = 0u;
    uint32_t fmax = 0u;
    std::array<uint32_t, 1024> lut{};
};


static DecoderData build_decoder_data(
    const Config& cfg
) {
    if (cfg.bc + cfg.bf != 32) {
        throw std::runtime_error("Bc+Bf != 32");
    }

    if (cfg.bf > 19) {
        throw std::runtime_error(
            "P3B LUT supports Bf <= 19"
        );
    }

    DecoderData result{};

    result.identity_coarse_mask =
        (1u << cfg.bc) - 1u;

    result.fmax =
        (1u << cfg.bf) - 1u;

    std::array<int, 32> inverse{};
    inverse.fill(-1);

    for (
        int logical_bit = 0;
        logical_bit < 32;
        ++logical_bit
    ) {
        const int physical_bit =
            cfg.mapping[
                static_cast<std::size_t>(logical_bit)
            ];

        if (
            physical_bit < 0
            ||
            physical_bit >= 32
        ) {
            throw std::runtime_error(
                "mapping physical bit outside 0..31"
            );
        }

        if (
            inverse[
                static_cast<std::size_t>(physical_bit)
            ]
            != -1
        ) {
            throw std::runtime_error(
                "duplicate physical mapping bit"
            );
        }

        inverse[
            static_cast<std::size_t>(physical_bit)
        ] = logical_bit;
    }

    for (int p = 0; p < 32; ++p) {
        if (
            inverse[
                static_cast<std::size_t>(p)
            ]
            < 0
        ) {
            throw std::runtime_error(
                "mapping inverse incomplete"
            );
        }
    }

    for (
        int byte_index = 0;
        byte_index < 4;
        ++byte_index
    ) {
        for (
            int byte_value = 0;
            byte_value < 256;
            ++byte_value
        ) {
            uint32_t fine_contribution = 0u;
            uint32_t coarse_count = 0u;

            for (
                int bit_in_byte = 0;
                bit_in_byte < 8;
                ++bit_in_byte
            ) {
                if (
                    ((byte_value >> bit_in_byte) & 1)
                    == 0
                ) {
                    continue;
                }

                const int physical_bit =
                    byte_index * 8 + bit_in_byte;

                const int logical_bit =
                    inverse[
                        static_cast<std::size_t>(
                            physical_bit
                        )
                    ];

                if (logical_bit < cfg.bc) {
                    ++coarse_count;
                }
                else {
                    const int fine_bit =
                        logical_bit - cfg.bc;

                    if (
                        fine_bit < 0
                        ||
                        fine_bit >= cfg.bf
                    ) {
                        throw std::runtime_error(
                            "invalid fine logical bit"
                        );
                    }

                    fine_contribution |=
                        1u << fine_bit;
                }
            }

            if (coarse_count > 8u) {
                throw std::runtime_error(
                    "coarse byte count > 8"
                );
            }

            const std::size_t index =
                static_cast<std::size_t>(
                    byte_index
                ) * 256ull
                +
                static_cast<std::size_t>(
                    byte_value
                );

            if (index >= result.lut.size()) {
                throw std::runtime_error(
                    "LUT index outside 0..1023"
                );
            }

            result.lut[index] =
                fine_contribution
                |
                (coarse_count << 24);
        }
    }

    return result;
}


static inline void decode_identity_metadata(
    uint32_t word,
    const Config& cfg,
    const DecoderData& decoder,
    uint32_t& coarse_count,
    uint32_t& fine
) {
    coarse_count =
        popcount32(
            word
            &
            decoder.identity_coarse_mask
        );

    fine =
        (word >> cfg.bc)
        &
        decoder.fmax;
}


static inline void decode_lut_metadata(
    uint32_t word,
    const DecoderData& decoder,
    uint32_t& coarse_count,
    uint32_t& fine
) {
    const uint32_t b0 =
        word & 0xFFu;

    const uint32_t b1 =
        (word >> 8) & 0xFFu;

    const uint32_t b2 =
        (word >> 16) & 0xFFu;

    const uint32_t b3 =
        (word >> 24) & 0xFFu;

    const uint32_t e0 =
        decoder.lut[
            static_cast<std::size_t>(b0)
        ];

    const uint32_t e1 =
        decoder.lut[
            256u
            +
            static_cast<std::size_t>(b1)
        ];

    const uint32_t e2 =
        decoder.lut[
            512u
            +
            static_cast<std::size_t>(b2)
        ];

    const uint32_t e3 =
        decoder.lut[
            768u
            +
            static_cast<std::size_t>(b3)
        ];

    fine =
        (e0 | e1 | e2 | e3)
        &
        decoder.fmax;

    coarse_count =
        ((e0 >> 24) & 0xFFu)
        +
        ((e1 >> 24) & 0xFFu)
        +
        ((e2 >> 24) & 0xFFu)
        +
        ((e3 >> 24) & 0xFFu);
}


static inline float scale_from_decoded(
    uint32_t coarse_count,
    uint32_t fine,
    const Config& cfg,
    const DecoderData& decoder
) {
    const double u =
        static_cast<double>(coarse_count)
        +
        static_cast<double>(fine)
        /
        static_cast<double>(decoder.fmax);

    const double z =
        cfg.zmin
        +
        (cfg.zmax - cfg.zmin)
        /
        static_cast<double>(cfg.bc + 1)
        *
        u;

    return static_cast<float>(
        std::exp2(z)
    );
}


static void run_identity(
    const std::vector<uint8_t>& packed,
    const std::vector<uint32_t>& logical_metadata,
    std::vector<float>& output,
    const Config& cfg,
    const DecoderData& decoder,
    int thread_count
) {
    const long long groups =
        static_cast<long long>(
            logical_metadata.size()
        );

    omp_set_num_threads(thread_count);

#pragma omp parallel for schedule(static)
    for (
        long long group_ll = 0;
        group_ll < groups;
        ++group_ll
    ) {
        const std::size_t group =
            static_cast<std::size_t>(
                group_ll
            );

        uint32_t coarse_count = 0u;
        uint32_t fine = 0u;

        decode_identity_metadata(
            logical_metadata[group],
            cfg,
            decoder,
            coarse_count,
            fine
        );

        const float scale =
            scale_from_decoded(
                coarse_count,
                fine,
                cfg,
                decoder
            );

        const uint8_t* packed_group =
            packed.data()
            +
            group
            *
            PACKED_BYTES_PER_GROUP;

        float* out_group =
            output.data()
            +
            group
            *
            GROUP_SIZE;

        for (
            int lane = 0;
            lane < 32;
            ++lane
        ) {
            out_group[
                static_cast<std::size_t>(lane)
            ] =
                static_cast<float>(
                    decode_q(
                        packed_group,
                        lane
                    )
                )
                *
                scale;
        }
    }
}


static void run_lut(
    const std::vector<uint8_t>& packed,
    const std::vector<uint32_t>& physical_metadata,
    std::vector<float>& output,
    const Config& cfg,
    const DecoderData& decoder,
    int thread_count
) {
    const long long groups =
        static_cast<long long>(
            physical_metadata.size()
        );

    omp_set_num_threads(thread_count);

#pragma omp parallel for schedule(static)
    for (
        long long group_ll = 0;
        group_ll < groups;
        ++group_ll
    ) {
        const std::size_t group =
            static_cast<std::size_t>(
                group_ll
            );

        uint32_t coarse_count = 0u;
        uint32_t fine = 0u;

        decode_lut_metadata(
            physical_metadata[group],
            decoder,
            coarse_count,
            fine
        );

        const float scale =
            scale_from_decoded(
                coarse_count,
                fine,
                cfg,
                decoder
            );

        const uint8_t* packed_group =
            packed.data()
            +
            group
            *
            PACKED_BYTES_PER_GROUP;

        float* out_group =
            output.data()
            +
            group
            *
            GROUP_SIZE;

        for (
            int lane = 0;
            lane < 32;
            ++lane
        ) {
            out_group[
                static_cast<std::size_t>(lane)
            ] =
                static_cast<float>(
                    decode_q(
                        packed_group,
                        lane
                    )
                )
                *
                scale;
        }
    }
}


static void consume_output(
    const std::vector<float>& output
) {
    if (output.empty()) {
        return;
    }

    double checksum = 0.0;

    const std::size_t step =
        std::max<std::size_t>(
            1,
            output.size() / 64
        );

    for (
        std::size_t index = 0;
        index < output.size();
        index += step
    ) {
        checksum +=
            static_cast<double>(
                output[index]
            );
    }

    GLOBAL_SINK += checksum;
}


template <typename Fn>
static double time_once_ms(
    Fn&& fn,
    const std::vector<float>& output
) {
    const auto start =
        std::chrono::steady_clock::now();

    fn();

    const auto stop =
        std::chrono::steady_clock::now();

    consume_output(output);

    return
        std::chrono::duration<
            double,
            std::milli
        >(
            stop - start
        ).count();
}


static void generate_workload(
    const Config& cfg,
    uint64_t seed,
    std::vector<uint8_t>& packed,
    std::vector<uint32_t>& logical_metadata,
    std::vector<uint32_t>& physical_metadata
) {
    packed.resize(
        GROUPS
        *
        PACKED_BYTES_PER_GROUP
    );

    logical_metadata.resize(GROUPS);
    physical_metadata.resize(GROUPS);

    std::mt19937_64 rng(seed);

    std::uniform_int_distribution<int> qdist(
        -7,
        7
    );

    std::uniform_real_distribution<double> zdist(
        cfg.zmin,
        cfg.zmax
    );

    for (
        std::size_t group = 0;
        group < GROUPS;
        ++group
    ) {
        uint8_t* packed_group =
            packed.data()
            +
            group
            *
            PACKED_BYTES_PER_GROUP;

        for (
            int pair_index = 0;
            pair_index < 16;
            ++pair_index
        ) {
            const int q0 = qdist(rng);
            const int q1 = qdist(rng);

            const uint8_t n0 =
                static_cast<uint8_t>(q0)
                &
                0x0Fu;

            const uint8_t n1 =
                static_cast<uint8_t>(q1)
                &
                0x0Fu;

            packed_group[
                static_cast<std::size_t>(
                    pair_index
                )
            ] =
                n0
                |
                static_cast<uint8_t>(
                    n1 << 4
                );
        }

        const double z =
            zdist(rng);

        const uint32_t logical =
            encode_sbl32(
                z,
                cfg
            );

        logical_metadata[group] =
            logical;

        physical_metadata[group] =
            logical_to_physical(
                logical,
                cfg
            );
    }
}


static bool correctness_test(
    const Config& cfg,
    const DecoderData& decoder,
    const std::vector<uint32_t>& logical_metadata,
    const std::vector<uint32_t>& physical_metadata,
    std::size_t& tested
) {
    if (
        logical_metadata.size()
        !=
        physical_metadata.size()
    ) {
        throw std::runtime_error(
            "metadata size mismatch"
        );
    }

    tested =
        std::min<std::size_t>(
            logical_metadata.size(),
            100000
        );

    if (tested == 0) {
        throw std::runtime_error(
            "zero correctness samples"
        );
    }

    for (
        std::size_t index = 0;
        index < tested;
        ++index
    ) {
        uint32_t id_count = 0u;
        uint32_t id_fine = 0u;

        uint32_t lut_count = 0u;
        uint32_t lut_fine = 0u;

        decode_identity_metadata(
            logical_metadata[index],
            cfg,
            decoder,
            id_count,
            id_fine
        );

        decode_lut_metadata(
            physical_metadata[index],
            decoder,
            lut_count,
            lut_fine
        );

        if (
            id_count != lut_count
            ||
            id_fine != lut_fine
        ) {
            return false;
        }
    }

    return true;
}


int main(
    int argc,
    char** argv
) {
    try {
        if (argc != 2) {
            std::cerr
                << "Usage: p3b_amd7840hs_multithread.exe <output_dir>\n";

            return 2;
        }

        const std::string output_dir =
            argv[1];

        const std::string raw_path =
            output_dir
            +
            "\\p3b_raw_blocks.csv";

        const std::string correctness_path =
            output_dir
            +
            "\\p3b_correctness.csv";

        omp_set_dynamic(0);

        std::cout
            << "BACKEND=AMD_Ryzen_7_7840HS_x86_64_CPU\n"
            << "OPENMP=True\n"
            << "OMP_DYNAMIC=False\n"
            << "THREAD_AFFINITY=OS_SCHEDULER_MANAGED\n"
            << "THREAD_COUNTS=1,4,8\n"
            << "GROUPS=4194304\n"
            << "GROUP_SIZE=32\n"
            << "TRUE_NIBBLE_PACKED_INT4=True\n"
            << "BYTE_LUT_BYTES=4096\n";


        std::ofstream raw_file(
            raw_path,
            std::ios::out
            |
            std::ios::trunc
        );

        if (!raw_file) {
            throw std::runtime_error(
                "cannot open raw CSV"
            );
        }

        raw_file
            << "representation_id,"
            << "Bc,"
            << "Bf,"
            << "threads,"
            << "groups,"
            << "block,"
            << "kernel,"
            << "ms\n";


        std::ofstream correctness_file(
            correctness_path,
            std::ios::out
            |
            std::ios::trunc
        );

        if (!correctness_file) {
            throw std::runtime_error(
                "cannot open correctness CSV"
            );
        }

        correctness_file
            << "representation_id,"
            << "Bc,"
            << "Bf,"
            << "groups,"
            << "tested_words,"
            << "lut_identity_exact\n";


        bool correctness_all = true;


        for (
            int config_index = 0;
            config_index < 4;
            ++config_index
        ) {
            const Config& cfg =
                CONFIGS[
                    static_cast<std::size_t>(
                        config_index
                    )
                ];

            const DecoderData decoder =
                build_decoder_data(cfg);

            std::cout
                << "\n"
                << "================================================================================\n"
                << "[P3B] "
                << cfg.id
                << " Bc="
                << cfg.bc
                << " Bf="
                << cfg.bf
                << "\n"
                << "================================================================================\n";


            std::vector<uint8_t> packed;
            std::vector<uint32_t> logical_metadata;
            std::vector<uint32_t> physical_metadata;

            generate_workload(
                cfg,
                20260902ull
                +
                static_cast<uint64_t>(
                    config_index
                )
                *
                100000ull,
                packed,
                logical_metadata,
                physical_metadata
            );


            std::size_t tested = 0;

            const bool correctness =
                correctness_test(
                    cfg,
                    decoder,
                    logical_metadata,
                    physical_metadata,
                    tested
                );

            correctness_all =
                correctness_all
                &&
                correctness;

            correctness_file
                << cfg.id
                << ","
                << cfg.bc
                << ","
                << cfg.bf
                << ","
                << GROUPS
                << ","
                << tested
                << ","
                << (
                    correctness
                    ? "True"
                    : "False"
                )
                << "\n";

            correctness_file.flush();

            if (!correctness) {
                throw std::runtime_error(
                    std::string(cfg.id)
                    +
                    ": LUT correctness failure"
                );
            }


            std::vector<float> output(
                GROUPS * GROUP_SIZE
            );


            for (int thread_count : THREAD_COUNTS) {

                std::cout
                    << "\n"
                    << "[THREADS="
                    << thread_count
                    << "]\n";


                auto identity_fn = [&]() {
                    run_identity(
                        packed,
                        logical_metadata,
                        output,
                        cfg,
                        decoder,
                        thread_count
                    );
                };


                auto lut_fn = [&]() {
                    run_lut(
                        packed,
                        physical_metadata,
                        output,
                        cfg,
                        decoder,
                        thread_count
                    );
                };


                for (
                    int warmup = 0;
                    warmup < WARMUP_RUNS;
                    ++warmup
                ) {
                    identity_fn();
                    lut_fn();
                }

                consume_output(output);


                for (
                    int block = 1;
                    block <= PAIRED_BLOCKS;
                    ++block
                ) {
                    double id_first = 0.0;
                    double id_second = 0.0;

                    double lut_first = 0.0;
                    double lut_second = 0.0;


                    if ((block & 1) != 0) {

                        id_first =
                            time_once_ms(
                                identity_fn,
                                output
                            );

                        lut_first =
                            time_once_ms(
                                lut_fn,
                                output
                            );

                        lut_second =
                            time_once_ms(
                                lut_fn,
                                output
                            );

                        id_second =
                            time_once_ms(
                                identity_fn,
                                output
                            );
                    }
                    else {

                        lut_first =
                            time_once_ms(
                                lut_fn,
                                output
                            );

                        id_first =
                            time_once_ms(
                                identity_fn,
                                output
                            );

                        id_second =
                            time_once_ms(
                                identity_fn,
                                output
                            );

                        lut_second =
                            time_once_ms(
                                lut_fn,
                                output
                            );
                    }


                    const double id_ms =
                        0.5
                        *
                        (
                            id_first
                            +
                            id_second
                        );

                    const double lut_ms =
                        0.5
                        *
                        (
                            lut_first
                            +
                            lut_second
                        );


                    if (
                        !std::isfinite(id_ms)
                        ||
                        !std::isfinite(lut_ms)
                        ||
                        id_ms <= 0.0
                        ||
                        lut_ms <= 0.0
                    ) {
                        throw std::runtime_error(
                            "invalid timing"
                        );
                    }


                    raw_file
                        << cfg.id
                        << ","
                        << cfg.bc
                        << ","
                        << cfg.bf
                        << ","
                        << thread_count
                        << ","
                        << GROUPS
                        << ","
                        << block
                        << ",ID_FULL,"
                        << std::setprecision(17)
                        << id_ms
                        << "\n";


                    raw_file
                        << cfg.id
                        << ","
                        << cfg.bc
                        << ","
                        << cfg.bf
                        << ","
                        << thread_count
                        << ","
                        << GROUPS
                        << ","
                        << block
                        << ",MAP_LUT8,"
                        << std::setprecision(17)
                        << lut_ms
                        << "\n";

                    raw_file.flush();


                    const double overhead =
                        100.0
                        *
                        (
                            lut_ms / id_ms
                            -
                            1.0
                        );


                    std::cout
                        << "[BLOCK "
                        << std::setw(2)
                        << block
                        << "/"
                        << PAIRED_BLOCKS
                        << "] "
                        << "ID="
                        << std::fixed
                        << std::setprecision(3)
                        << id_ms
                        << "ms "
                        << "LUT="
                        << lut_ms
                        << "ms "
                        << "LUTvsID="
                        << std::showpos
                        << overhead
                        << "%"
                        << std::noshowpos
                        << "\n";
                }
            }
        }


        raw_file.close();
        correctness_file.close();


        std::cout
            << "\nCORRECTNESS_ALL="
            << (
                correctness_all
                ? "True"
                : "False"
            )
            << "\n";

        std::cout
            << "GLOBAL_SINK="
            << std::setprecision(17)
            << GLOBAL_SINK
            << "\n";

        std::cout
            << "P3B_CPP_PASS=True\n";


        return
            correctness_all
            ? 0
            : 3;
    }
    catch (const std::exception& ex) {

        std::cerr
            << "FATAL: "
            << ex.what()
            << "\n";

        return 10;
    }
}

