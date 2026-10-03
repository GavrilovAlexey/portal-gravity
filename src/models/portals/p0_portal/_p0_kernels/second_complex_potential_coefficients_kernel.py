import cupy as cp

second_complex_potential_coefficients_kernel = cp.ElementwiseKernel(
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
            term *= X(0, 1) / length;
            if (is_close) {
                term = X(0, term.imag());
            }
            return term;
        };

        complex_potential_coefficient += calculate_term(point, first_start, first_end, TOLERANCE);
        complex_potential_coefficient += calculate_term(point, second_start, second_end, TOLERANCE) *
            (is_inverted ? X(1.0, 0.0) : X(-1.0, 0.0));
    """,
    name="second_complex_potential_coefficients_kernel"
)