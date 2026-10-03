import cupy as cp

first_field_strength_coefficients_kernel = cp.ElementwiseKernel(
    in_params="X point, X first_start, X first_end, X second_start, X second_end, bool is_inverted, float TOLERANCE",
    out_params="X complex_potential_coefficient",
    operation="""
        auto calculate_term = [](X point, X start, X end, float TOLERANCE) -> X {
            float length = abs(end - start);
            X normalized_direction = (end - start) /  length;

            X local_point = (point - start) * conj(normalized_direction);
            bool is_close = (0.0 <= local_point.real()) && 
                            (local_point.real() <= length) && 
                            (abs(local_point.imag()) <= TOLERANCE);

            X term = log(X(1, 0) - length / local_point);
            term = conj(term) / length;
            if (is_close) {
                term = X(term.real(), 0);
            }
            return term * normalized_direction;
        };

        complex_potential_coefficient += calculate_term(point, first_start, first_end, TOLERANCE);
        complex_potential_coefficient -= calculate_term(point, second_start, second_end, TOLERANCE);
    """,
    name="first_field_strength_coefficients_kernel"
)
