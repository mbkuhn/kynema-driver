#ifndef DRIVER_CFD_MODULE_ADAPTER_H
#define DRIVER_CFD_MODULE_ADAPTER_H

#include "ModuleAdapter.h"

namespace driver {

// Keep the Tioga hooks separate from the common lifecycle and JSON interface.
template <typename NativeModule, bool BackgroundOnly>
class CfdModuleAdapter final : public ModuleAdapter<NativeModule>
{
public:
    using ModuleAdapter<NativeModule>::ModuleAdapter;

    bool is_cfd() override { return true; }
    bool is_background_only() override { return BackgroundOnly; }
    int overset_update_interval() override
    {
        return this->native().overset_update_interval();
    }
    int get_ncomps() override
    {
        if constexpr (BackgroundOnly)
            return Module::get_ncomps();
        else
            return this->native().get_ncomps();
    }
    void pre_overset_conn_work() override
    {
        this->native().pre_overset_conn_work();
    }
    void post_overset_conn_work() override
    {
        this->native().post_overset_conn_work();
    }
    // This also finishes native pre-advance work when Tioga is disabled.
    // Keep it independent of the optional connectivity and field-transfer hooks.
    void prepare_solution_exchange() override
    {
        this->native().prepare_solution_exchange();
    }
    void register_solution() override { this->native().register_solution(); }
    void update_solution() override { this->native().update_solution(); }
};

template <typename NativeModule>
struct CfdModuleAdapterTraits
{
    static bool is_fixed_timestep_size(NativeModule& module)
    {
        return module.is_fixed_timestep_size();
    }
    static PreparationStage
    prepare(NativeModule& module, const PreparationContext& context)
    {
        return module.prepare(context);
    }
};

} // namespace driver
#endif
