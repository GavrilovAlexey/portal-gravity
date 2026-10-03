import cupy as cp

field_strengths_kernel = cp.ElementwiseKernel(
    in_params="""
        X point,
        raw X first_starts, raw X first_ends,
        raw X second_starts, raw X second_ends,
        bool is_inverted,
        raw T parameters,
        T TOLERANCE
    """,
    out_params="X field_strength",
    operation="""
        auto calculate_first_term = [](X local_point, T length, bool is_close, X normalized_direction) -> X {
            X term = log(X(1, 0) - length / local_point);
            term = conj(term) / length;   

            if (is_close) {
                term = X(term.real(), 0);
            }
            return term * normalized_direction;
        };

        auto calculate_second_term = [](X local_point, T length, bool is_close, X normalized_direction) -> X {
            X term = (length - local_point) * local_point;
            term = X(0, 1) / conj(term);

            return term * normalized_direction;
        };


        int n = first_starts.size();
        X local_point, normalized_direction;
        T length;
        bool is_close;
        
        for (int j = 0; j < n; j++) {
            length = abs(first_ends[j] - first_starts[j]);
            normalized_direction = (first_ends[j] - first_starts[j]) /  length;

            local_point = (point - first_starts[j]) * conj(normalized_direction);
            is_close = (0.0 <= local_point.real()) && 
                       (local_point.real() <= length) && 
                       (abs(local_point.imag()) <= TOLERANCE);

            field_strength += calculate_first_term(local_point, length, is_close, normalized_direction) *
                parameters[j];
            field_strength += calculate_second_term(local_point, length, is_close, normalized_direction) *
                parameters[n + j];           


            length = abs(second_ends[j] - second_starts[j]);
            normalized_direction = (second_ends[j] - second_starts[j]) / length;

            local_point = (point - second_starts[j]) * conj(normalized_direction);
            is_close = (0.0 <= local_point.real()) && 
                       (local_point.real() <= length) && 
                       (abs(local_point.imag()) <= TOLERANCE);

            field_strength += calculate_first_term(local_point, length, is_close, normalized_direction) *
                -parameters[j];
            field_strength += calculate_second_term(local_point, length, is_close, normalized_direction) *
                parameters[n + j] * (is_inverted ? X(1.0, 0.0) : X(-1.0, 0.0));
        }
    """,
    name="field_strengths_kernel"
)