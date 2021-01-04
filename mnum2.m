data = [
    -5 -5.4606;
    -4 -3.8804;
    -3 -1.9699;
    -2 -1.6666;
    -1 -0.0764;
    0 -0.3971;
    1 -1.0303;
    2 -4.5483;
    3 -11.528;
    4 -21.6417;
    5 -34.4458
];

N = size(data(:, 1), 1);
max_poly_order = 6;
x = data(:, 1);
y = data(:, 2);
A = zeros(N, max_poly_order);
% data used to plot approximated functions
approx_x = [-5 : 0.1 : 5]';
for i = 0:max_poly_order
    % A matrix init for given step
    A(:, i+1) = x.^(i);
    current_A = A(:, 1:i+1);
    % using normal equations
    norm_eq_a = normal_equations_least_squares(current_A, y);
    norm_eq_residue = current_A*norm_eq_a - y;
    fprintf('Norma residuum dla równań normalnych i wielomianu stopnia %d: %d\n', i, norm(norm_eq_residue));
    norm_eq_y = polyval(norm_eq_a, approx_x);
    norm_eq_samples_approximation_error = norm(polyval(norm_eq_a, x) - y);
    fprintf('Norma błędów dla próbek: %d\n', norm_eq_samples_approximation_error);
    % using QR decomposition
    qr_decomp_a = qr_decomposition_least_squares(current_A, y);
    qr_decomp_residue = current_A*qr_decomp_a - y;
    fprintf('Norma residuum dla rozkładu QR i wielomianu stopnia %d: %d\n', i, norm(qr_decomp_residue));
    qr_decomp_y = polyval(qr_decomp_a, approx_x);
    qr_decomp_samples_approximation_error = norm(polyval(qr_decomp_a, x) - y);
    fprintf('Norma błędów dla próbek: %d\n', qr_decomp_samples_approximation_error);
    % plot data and approximation
    figure('Name', "Row. norm. st. " + i, 'NumberTitle', 'off')
    plot(x, y, 'o')
    title("Metoda równań normalnych, wielomian st. " + i);
    hold on;
    plot(approx_x, norm_eq_y)
    hold off;
%     saveas(gcf, "./plots/normal eq/normal_eq_" + i + ".png");
    figure('Name', "Rozkl. QR st. " + i, 'NumberTitle', 'off')
    plot(x, y, 'o')
    title("Rozkład QR, wielomian st. " + i);
    hold on;
    plot(approx_x, qr_decomp_y)
    hold off;
%     saveas(gcf, "./plots/qr decomp/qr_decomp_" + i + ".png");
end

function coefficients = normal_equations_least_squares(A, b)
    gram_matrix = A'*A;
    fprintf('####################################\nRównania normalne\nWsk. uwarunkowania A: %d\n', get_condition_number(gram_matrix));
    coefficients = (gram_matrix)\(A'*b);
    coefficients = flipud(coefficients);
end

function coefficients = qr_decomposition_least_squares(A, b)
    [q, r] = qr_decomposition(A);
    fprintf('####################################\nRozkład QR\nWsk. uwarunkowania R: %d\n', get_condition_number(r));
    coefficients = r\(q'*b);
    coefficients = flipud(coefficients);
end

function [Q, R] = qr_decomposition(A)
    % QR decomposition using modified Gramm-Schmidt algorithm
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    d = zeros(1, n);
    for i = 1:n
        Q(:, i) = A(:, i);
        R(i, i) = 1;
        d(i) = Q(:, i)'*Q(:, i);
        % algorithm modification
        for j = i+1:n
            R(i, j) = (Q(:, i)'*A(:, j))/d(i);
            A(:, j) = A(:, j) - R(i, j)*Q(:, i);
        end    
    end
    % normalization
    for i = 1:n
        dd = norm(Q(:, i));
        Q(:, i) = Q(:, i)/dd;
        R(i, i:n) = R(i, i:n)*dd;
    end    
end

function condition_number = get_condition_number(A)
    max_norm = 0;
    max_inv_norm = 0;
    inv_matrix = A^(-1);
    rows_number = size(A, 1);
    for i = 1:rows_number
        current_norm = sum(abs(A(i, :)));
        if current_norm > max_norm
            max_norm = current_norm;
        end
        current_inv_norm = sum(abs(inv_matrix(i, :)));
        if current_inv_norm > max_inv_norm
            max_inv_norm = current_inv_norm;
        end
    end
    condition_number = max_norm * max_inv_norm;
end