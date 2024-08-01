#include "../include/quotients.cuh"
#include <cstdio>

void accumulate_quotients(
        point domain_initial_point,
        point domain_step,
        uint32_t domain_size,
        uint32_t **columns,
        uint32_t number_of_columns,
        qm31 random_coeff,
        secure_field_point *sample_points,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3
) {
    printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
    printf("%d\n", domain_size);
    printf("%d\n", sample_points[0].x.a.a);
    printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
}
