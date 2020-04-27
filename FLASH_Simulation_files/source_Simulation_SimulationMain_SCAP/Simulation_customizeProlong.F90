!!****if* source/Simulation/SimulationMain/LaserSlab/Simulation_customizeProlong
!!
!! NAME
!!
!!  Simulation_customizeProlong
!!
!! SYNOPSIS
!!
!!  call Simulation_customizeProlong(integer(IN)  :: beforeOrAfter)
!!
!! DESCRIPTION
!!
!!  The Simulation_customizeProlong interface provides a way to
!!  customize the prolongation of Grid data that normally happens
!!  after an AMR Grid has changed - in particular, the interpolation
!!  of data into blocks that were newly created by refining existing
!!  blocks.
!!
!!  After the refinement, apply the user-defined routine for a 
!!  special prolongation routine from the coarse parents to 
!!  the newly created child blocks.
!!
!!  The interface is called twice for each time that the global
!!  prolongation operation is applied: once just before prolongation
!!  gets applied, and then again after prolongation is done.
!!  The single argument beforeOrAfter is used to distinguish the calls.
!!
!!  This implementation affects only the prolongation of face variables.
!!  It leaves the handling of cell-centered variables unchanged.
!!
!!  Currently there are two divergence-free prolongation algorithms available:
!!  First is to use the "injection method" in which the face-centered 
!!  divergence-free magnetic fields of the parent blocks are directly 
!!  injected onto the newly born children blocks without any spatial 
!!  interpolation, therefore, automatically maintain divergence-free
!!  properties on the refined blocks. This method works fine in many cases,
!!  especially for smoothly varying fields. It is available for both 
!!  2D and 3D calculation.
!!  Second is the "Balsara's method" that uses bilinear polynomials that
!!  prolong the divergence-free facevars of the parent block data
!!  to the newly born favevars of the children block data, keeping the
!!  divergence-free properties.
!!  See Paramesh's users guide for more detail.
!!
!! ARGUMENTS
!!
!!   beforeOrAfter : BEFORE when called before prolongation,
!!                   AFTER when called after prolongation.
!!
!!
!! EXAMPLE  
!!
!!  This is how this interface is normally called, in one of the
!!  subroutines that implement Grid_updateRefinement processing:
!!
!!    #include "constants.h" 
!!    #include "Flash.h" 
!!    ...
!!       use Grid_data,ONLY: gr_meshMe            ! my rank
!!    ...
!!       ! call before to modify prolongation method:
!!       call Simulation_customizeProlong(BEFORE)
!!       ! PARAMESH routine that does prolongation:
!!       call amr_prolong (gr_meshMe, 1, NGUARD)
!!       ! call after to restore prolongation method:
!!       call Simulation_customizeProlong(AFTER)
!!    ...
!!
!! NOTES
!!
!!   The constants BEFORE and AFTER are defined in constants.h.
!!
!!   Since FLASH 3.1.1, a non-stub implementation for use by
!!   MHD simulations is provided in the directory tree location
!!   source/Simulation/SimulationMain/magnetoHD . All simulations
!!   placed under the magnetoHD directory will therefore use this
!!   implementation by default.  This is usually desired for
!!   simulations using MHD.  For this reason, simulations using
!!   MHD should be placed under the magnetoHD directory.
!!
!!***

subroutine Simulation_customizeProlong(beforeOrAfter)

#include "Flash.h"

#ifdef FLASH_GRID_PARAMESH

#ifdef LIBRARY
#define LIBRARY_OR_PM4DEV
#else
#ifdef FLASH_GRID_PARAMESH4DEV
#define LIBRARY_OR_PM4DEV
#endif
#endif

#if NFACE_VARS > 0
  use physicaldata, ONLY : interp_mask_facex,interp_mask_facey,interp_mask_facez, &
       force_consistency
  use paramesh_dimensions, ONLY : nfacevar
  use Hydro_data, ONLY : hy_prol_method
  use paramesh_interfaces, ONLY : prol_fc_dbz_init
#endif
#endif

  use Driver_interface,ONLY : Driver_abortFlash

  implicit none
#include "constants.h"
#if NFACE_VARS > 0
#include "UHD.h"
#endif
  integer, intent(IN) :: beforeOrAfter ! BEFORE for before, AFTER for after

#ifdef FLASH_GRID_PARAMESH
#if NFACE_VARS > 0
  integer, parameter :: totFaces = 3
  integer, dimension(NFACE_VARS),save :: interp_mask_facex_old,&
       interp_mask_facey_old,&
       interp_mask_facez_old
  logical,save :: forceConsistencyOld
  logical,save :: firstTime = .TRUE.
  integer, dimension(totFaces,nfacevar) :: mhd_divb_fc_vars
  integer :: i
#endif
#endif

  if (beforeOrAfter == BEFORE) then

#ifdef FLASH_GRID_PARAMESH
#if NFACE_VARS > 0
     interp_mask_facex_old(:)=interp_mask_facex(:)
     interp_mask_facey_old(:)=interp_mask_facey(:)
     interp_mask_facez_old(:)=interp_mask_facez(:)
     forceConsistencyOld = force_consistency
     if (firstTime) then
        call initProlMethod(hy_prol_method)
        firstTime = .FALSE.
     end if

     if (hy_prol_method == INJECTION_PROL) then
        !! zeroth order
        interp_mask_facex(:)=0
        interp_mask_facey(:)=0
        interp_mask_facez(:)=0

     elseif (hy_prol_method == BALSARA_PROL) then
     !! At this point we assume there are "nfacevar" families
     !! of divergenceless fields. 
     !! i.e., all facevars are assumed to be divergenceless.
        do i = 1,nfacevar
           mhd_divb_fc_vars(1:totFaces,i) = i
        enddo
        call prol_fc_dbz_init(nfacevar, mhd_divb_fc_vars)

     else
        call Driver_abortFlash&
             ("[Grid_updateRefinement]: unknown prolongation algorithm for face-centered magnetic fields!")
     endif
#ifdef LIBRARY_OR_PM4DEV
     force_consistency = .FALSE.
#endif  
#endif  
#endif

  else if (beforeOrAfter == AFTER) then

#ifdef FLASH_GRID_PARAMESH
#if NFACE_VARS > 0
     interp_mask_facex(:)=interp_mask_facex_old(:)
     interp_mask_facey(:)=interp_mask_facey_old(:)
     interp_mask_facez(:)=interp_mask_facez_old(:)
#ifdef LIBRARY_OR_PM4DEV
     force_consistency = forceConsistencyOld
#endif
#endif
#endif


  else
        
     call Driver_abortFlash("Simulation_customizeProlong: this is meaningless!")
  end if
     

#ifdef FLASH_USM_MHD
#if NFACE_VARS > 0
contains
  subroutine initProlMethod(prolMethod)
    use RuntimeParameters_interface, ONLY : RuntimeParameters_get
    use Hydro_data, ONLY : hy_prol_method_str

    integer,intent(INOUT) :: prolMethod

    if (prolMethod .NE. INJECTION_PROL .AND. &
        prolMethod .NE. BALSARA_PROL) then
       ! Duplicate this initialization from Hydro_init, since it is possible
       ! that Simulation_customizeProlong gets called before Hydro_init!
       ! We reuse hy_prol_method_str string variable from the Hydro unit.
       call RuntimeParameters_get('prolMethod',          hy_prol_method_str)
       if(trim(hy_prol_method_str) == "injection_prol" .or. &
            trim(hy_prol_method_str) == "INJECTION_PROL" ) then
          prolMethod = INJECTION_PROL
       elseif (trim(hy_prol_method_str) == "balsara_prol" .or. &
            trim(hy_prol_method_str) == "BALSARA_PROL" ) then
          prolMethod = BALSARA_PROL
       end if
    end if
  end subroutine initProlMethod
#endif
#endif

end subroutine Simulation_customizeProlong
