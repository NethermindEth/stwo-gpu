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
        uint32_t *sample_column_indexes,
        qm31 *sample_column_values,
        uint32_t *sample_column_and_values_sizes,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3
) {
    printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
    printf("%d\n", domain_size);
    printf("%d\n", sample_points[0].x.a.a);
    printf("%d %d\n", sample_column_indexes[0], sample_column_indexes[1]);
    printf("%d\n", sample_column_and_values_sizes[0]);
    printf("%d %d %d %d\n", sample_column_values[0].a.a, sample_column_values[0].a.b, sample_column_values[0].b.a, sample_column_values[0].b.b);
    printf("%d %d %d %d\n", sample_column_values[1].a.a, sample_column_values[1].a.b, sample_column_values[1].b.a, sample_column_values[1].b.b);
    printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
}
