#include "src/Simulation.h"

#include <stdexcept>
#include <string>
#include <vector>

class TestModule : public driver::Module
{
public:
    explicit TestModule(bool cfd = false) : is_cfd_module(cfd) {}

    std::vector<std::string> calls;
    int steps{0};
    bool is_cfd_module;

    bool is_cfd() override { return is_cfd_module; }
    bool is_fixed_timestep_size() override { return true; }
    int time_index() override { return steps; }
    driver::PreparationStage prepare(const driver::PreparationContext&) override
    {
        return driver::PreparationStage::complete;
    }
    double get_time() override { return steps; }
    double get_timestep_size() override { return 1.; }
    void set_timestep_size(double) override {}
    void finalize_time_step() override
    {
        calls.push_back("finalize");
        ++steps;
    }
    void dump_simulation_time() override {}
    void prepare_solution_exchange() override { calls.push_back("exchange"); }
    void set_inputs(std::size_t iteration) override
    {
        calls.push_back("set" + std::to_string(iteration));
    }
    void predict_inputs(std::size_t iteration) override
    {
        calls.push_back("predict" + std::to_string(iteration));
    }
    void reset_states() override { calls.push_back("reset"); }
    void advance_states(std::size_t iteration) override
    {
        calls.push_back("advance" + std::to_string(iteration));
    }
    void calculate_outputs(std::size_t iteration) override
    {
        calls.push_back("output" + std::to_string(iteration));
    }
    void additional_picard_iterations(int count) override
    {
        for (int i = 0; i < count; ++i) {
            reset_states();
            advance_states(static_cast<std::size_t>(i));
        }
        calculate_outputs(0);
    }
};

class BatchCorrectionModule : public TestModule
{
public:
    void additional_picard_iterations(int count) override
    {
        calls.push_back("batch" + std::to_string(count));
    }
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        driver::Simulation sim(MPI_COMM_WORLD);
        auto& corrected = sim.register_module_no_tioga<TestModule>(MPI_COMM_WORLD);
        auto& unchanged = sim.register_module_no_tioga<TestModule>(MPI_COMM_WORLD, true);
        auto& batched = sim.register_module_no_tioga<BatchCorrectionModule>(MPI_COMM_WORLD);
        if (rank == 0) {
            sim.set_additional_picard_iterations(corrected, 2);
            sim.set_additional_picard_iterations(batched, 3);
        }
        sim.initialize();
        corrected.calls.clear();
        unchanged.calls.clear();
        batched.calls.clear();
        sim.run_timesteps(2, 2);

        const std::vector<std::string> expected_step{
            "set0", "predict0", "advance0", "output0",
            "set1", "predict1", "reset", "advance1", "output1",
            "reset", "advance0", "reset", "advance1", "output0", "finalize"};
        const std::vector<std::string> unchanged_step{
            "set0", "predict0", "exchange", "advance0", "output0",
            "set1", "predict1", "exchange", "reset", "advance1", "output1",
            "exchange", "finalize"};
        const std::vector<std::string> no_correction_step{
            "set0", "predict0", "advance0", "output0",
            "set1", "predict1", "reset", "advance1", "output1", "finalize"};
        const auto& local_step = rank == 0 ? expected_step : no_correction_step;
        auto expected = local_step;
        expected.insert(expected.end(), local_step.begin(), local_step.end());
        auto uncorrected = unchanged_step;
        uncorrected.insert(uncorrected.end(), unchanged_step.begin(), unchanged_step.end());
        auto batch_step = no_correction_step;
        if (rank == 0) batch_step.insert(batch_step.end() - 1, "batch3");
        auto batch_calls = batch_step;
        batch_calls.insert(batch_calls.end(), batch_step.begin(), batch_step.end());
        if (corrected.calls != expected || corrected.steps != 2 ||
            unchanged.calls != uncorrected || unchanged.steps != 2 ||
            batched.calls != batch_calls || batched.steps != 2)
            throw std::runtime_error("Per-module Picard lifecycle mismatch");
    }
    {
        driver::Simulation sim(MPI_COMM_WORLD);
        auto& module = sim.register_module_no_tioga<TestModule>(MPI_COMM_WORLD, true);
        sim.initialize();
        module.calls.clear();
        sim.run_timesteps(1, 1);
        const std::vector<std::string> expected{
            "set0", "predict0", "exchange", "advance0", "output0", "finalize"};
        if (module.calls != expected)
            throw std::runtime_error("Unexpected exchange without Picard corrections");
    }
    MPI_Finalize();
}
