#ifndef KYNEMAFMB_H
#define KYNEMAFMB_H

#include "ModuleAdapter.h"
#include "interfaces/generalized/generalized_structure_interface.hpp"

namespace driver {

template <>
struct ModuleAdapterTraits<kynema_fmb::interfaces::GeneralStructureInterface>
{
    static constexpr bool fixed_timestep = true;

    static PreparationStage prepare(
        kynema_fmb::interfaces::GeneralStructureInterface& structure,
        const PreparationContext&)
    {
        structure.prepare();
        return PreparationStage::complete;
    }
};

using KynemaFMB =
    ModuleAdapter<kynema_fmb::interfaces::GeneralStructureInterface>;

} // namespace driver
#endif
