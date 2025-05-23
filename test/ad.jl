using ForwardDiff
using FiniteDiff

@testset "Check NeuralFoil Derivatives" begin
    function wrapfun(var)
        x, y = naca4(var[1], var[2], var[3])
        coordinates = [x y]

        outputs = get_aero_from_coordinates(
            coordinates, range(-5, 5, 3); reynolds=1e6, model_size="xlarge"
        )

        return [outputs.cl; outputs.cd; outputs.cm; outputs.confidence]
    end

    nacaparam = [2.0, 4.0, 12.0]
    adjac = ForwardDiff.jacobian(wrapfun, nacaparam)
    fdjac = FiniteDiff.finite_difference_jacobian(wrapfun, nacaparam)

    @test isapprox(adjac, fdjac, atol=1e-7)
end
