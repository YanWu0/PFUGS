function guess_list = fun_build_guess_list_around_value(value, q, base)
% fun_build_guess_list_around_value
% Build a geometric guess list centered around a positive reference value.
% If value = m * base^e, we round m to one decimal and return
%   m_rounded * base^(e-q), ..., m_rounded * base^(e+q).

    if value <= 0
        error('The reference value must be positive.');
    end
    if q < 0 || abs(q - round(q)) > 1e-12
        error('q must be a nonnegative integer.');
    end
    if nargin < 3 || isempty(base)
        base = 10;
    end
    if base <= 1
        error('base must be greater than 1.');
    end

    exponent_center = floor(log(value) / log(base));
    mantissa = value / (base^exponent_center);
    mantissa_rounded = round(10 * mantissa) / 10;

    exponent_list = (exponent_center - q):(exponent_center + q);
    guess_list = mantissa_rounded * (base .^ exponent_list);
end
