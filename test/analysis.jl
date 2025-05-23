#---------------------------------#
#        Wrap Python Version      #
#---------------------------------#

using PythonCall
using CondaPkg
CondaPkg.add_pip("neuralfoil")
CondaPkg.add_pip("aerosandbox")

__precompile__(false)

macro wrap_pyfuns(modsym, fname, cname)
    quote
        const pymod = pyimport($modsym)
        # Define functions with the same names as the input symbols
        $(:(function $(esc(cname))(args...; kwargs...)
            pyf = @pyconst pymod.$(fname)
            return pyf(args...; kwargs...)
        end))
    end
end

@wrap_pyfuns "neuralfoil" get_aero_from_coordinates get_aero_from_coordinates_py
@wrap_pyfuns "aerosandbox.geometry.airfoil.airfoil_families" get_kulfan_parameters get_kulfan_parameters_py
@wrap_pyfuns "numpy" array np_array

function wrapped_nf(coordinates, alpha, reynolds; model_size="xlarge")
    aero = get_aero_from_coordinates_py(
        np_array(coordinates); alpha=alpha, Re=reynolds, model_size=model_size
    )
    return (;
        cl=pyconvert(Vector{Float64}, aero["CL"]),
        cd=pyconvert(Vector{Float64}, aero["CD"]),
        cm=pyconvert(Vector{Float64}, aero["CM"]),
        analysis_confidence=pyconvert(Vector{Float64}, aero["analysis_confidence"]),
    )
end

#---------------------------------#
#            Run Tests            #
#---------------------------------#

@testset "Compare Kulfan Solvers" begin
    coordinates = NeuralFoil.get_coordinates_from_file("data/naca_2412_blunt.dat")

    cst_outs = get_kulfan_parameters_py(np_array(coordinates))
    cst_lower_nf = pyconvert(Vector{Float64}, cst_outs["lower_weights"])
    cst_upper_nf = pyconvert(Vector{Float64}, cst_outs["upper_weights"])
    cst_TE_nf = pyconvert(Float64, cst_outs["TE_thickness"])
    cst_LE_nf = pyconvert(Float64, cst_outs["leading_edge_weight"])

    kulfan_parameters = NeuralFoil.get_kulfan_parameters(coordinates)

    @test isapprox(kulfan_parameters.upper_weights, cst_upper_nf, atol=1e-8)
    @test isapprox(kulfan_parameters.lower_weights, cst_lower_nf, atol=1e-8)
    @test isapprox(kulfan_parameters.leading_edge_weight[], cst_LE_nf, atol=1e-8)
    @test isapprox(kulfan_parameters.TE_thickness[], cst_TE_nf, atol=1e-8)
end

@testset "Compare to NeuralFoil" begin
    coordinates = NeuralFoil.get_coordinates_from_file("data/naca_2412_blunt.dat")
    alpha = range(-5, 15; step=1)
    model_size = "xlarge"

    # single reynolds test
    reynolds = 1e6

    outputs_nfpy = wrapped_nf(coordinates, alpha, reynolds; model_size=model_size)

    outputs_nfjl = get_aero_from_coordinates(
        coordinates, alpha, reynolds; model_size=model_size
    )

    @test isapprox(outputs_nfpy.cl, outputs_nfjl.cl, atol=1e-8)
    @test isapprox(outputs_nfpy.cd, outputs_nfjl.cd, atol=1e-8)
    @test isapprox(outputs_nfpy.cm, outputs_nfjl.cm, atol=1e-8)
    @test isapprox(
        outputs_nfpy.analysis_confidence, outputs_nfjl.analysis_confidence, atol=1e-8
    )

    # multiple reynolds test (check for desired shapes)
    reynolds = [1e6; 2e6]

    outputs_nfjl = get_aero_from_coordinates(
        coordinates, alpha, reynolds; model_size=model_size
    )

    @test size(outputs_nfjl.cl) == (length(alpha), length(reynolds))
    @test size(outputs_nfjl.upper_bl_ue_over_vinf, 1) == length(alpha)
    @test size(outputs_nfjl.upper_bl_ue_over_vinf, 3) == length(reynolds)
end
