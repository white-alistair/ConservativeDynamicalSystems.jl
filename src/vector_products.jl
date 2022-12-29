function dot_product(a, b)
    return sum(a .* b)
end

⋅(a, b) = dot_product(a, b)

function cross_product(a, b)
    a₁, a₂, a₃ = a
    b₁, b₂, b₃ = b

    return [
        a₂ * b₃ - a₃ * b₂,
        a₃ * b₁ - a₁ * b₃,
        a₁ * b₂ - a₂ * b₁,
        ]
end

×(a, b) = cross_product(a, b)
