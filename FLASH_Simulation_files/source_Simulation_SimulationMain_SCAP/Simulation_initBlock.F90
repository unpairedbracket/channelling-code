!!****if* source/Simulation/SimulationMain/LaserSlab/Simulation_initBlock
!!
!! NAME
!!
!!  Simulation_initBlock
!!
!!
!! SYNOPSIS
!!
!!  call Simulation_initBlock(integer(IN) :: blockID) 
!!                       
!!
!!
!!
!! DESCRIPTION
!!
!!  Initializes fluid data (density, pressure, velocity, etc.) for
!!  a specified block.
!! 
!! ARGUMENTS
!!
!!  blockID -        the number of the block to initialize
!!  
!!
!!
!!***


subroutine Simulation_initBlock(blockId)
  use Simulation_data
  use Grid_interface, ONLY : Grid_getBlkIndexLimits, &
       Grid_getCellCoords, Grid_putPointData,&
                           Grid_getBlkPtr,         &
                           Grid_releaseBlkPtr

  use Driver_interface, ONLY: Driver_abortFlash
  use RadTrans_interface, ONLY: RadTrans_mgdEFromT

  implicit none

#include "constants.h"
#include "Flash.h"

  ! compute the maximum length of a vector in each coordinate direction 
  ! (including guardcells)

  integer, intent(in) :: blockId
  
  integer :: i, j, k, n
  integer :: blkLimits(2, MDIM)
  integer :: blkLimitsGC(2, MDIM)
  integer :: axis(MDIM)
  real, allocatable :: xcent(:), ycent(:), zcent(:)
  real :: tradActual, r2, sigma2
  real :: rho, tele, trad, tion, zbar, abar
  real :: targ_fraction, cham_fraction
  real, pointer, dimension(:,:,:,:) :: facexData,faceyData
#if NDIM > 0
  real, pointer, dimension(:,:,:,:) :: facezData
#endif

#ifndef CHAM_SPEC
  integer :: CHAM_SPEC = 1, TARG_SPEC = 2
#endif


  ! get the coordinate information for the current block from the database
  call Grid_getBlkIndexLimits(blockId,blkLimits,blkLimitsGC)

  ! get the coordinate information for the current block from the database
  call Grid_getBlkIndexLimits(blockId,blkLimits,blkLimitsGC)
  allocate(xcent(blkLimitsGC(HIGH, IAXIS)))
  call Grid_getCellCoords(IAXIS, blockId, CENTER, .true., &
       xcent, blkLimitsGC(HIGH, IAXIS))
  allocate(ycent(blkLimitsGC(HIGH, JAXIS)))
  call Grid_getCellCoords(JAXIS, blockId, CENTER, .true., &
       ycent, blkLimitsGC(HIGH, JAXIS))
  allocate(zcent(blkLimitsGC(HIGH, KAXIS)))
  call Grid_getCellCoords(KAXIS, blockId, CENTER, .true., &
       zcent, blkLimitsGC(HIGH, KAXIS))

#if NFACE_VARS > 0
  if (sim_killdivb) then
     call Grid_getBlkPtr(blockID,facexData,FACEX)
     call Grid_getBlkPtr(blockID,faceyData,FACEY)
     if (NDIM>2) call Grid_getBlkPtr(blockID,facezData,FACEZ)
  endif
#endif

  sigma2 = sim_targetSigma**2.0

  !------------------------------------------------------------------------------

  ! Loop over cells and set the initial state
  do k = blkLimits(LOW,KAXIS),blkLimits(HIGH,KAXIS)
     do j = blkLimits(LOW,JAXIS),blkLimits(HIGH,JAXIS)
        do i = blkLimits(LOW,IAXIS),blkLimits(HIGH,IAXIS)

           axis(IAXIS) = i
           axis(JAXIS) = j
           axis(KAXIS) = k

           targ_fraction = 1

           if (sim_initGeom == "slab") then
             if ( xcent(i) > 0 ) then
               targ_fraction = targ_fraction * &
                 EXP(-(xcent(i))**2.0 /sigma2)
             elseif ( xcent(i) < -sim_targetThickness ) then
               targ_fraction = targ_fraction * &
                 EXP(-(xcent(i) + sim_targetThickness)**2.0 / sigma2)
             endif
             if ( abs(ycent(j)) > sim_targetRadius ) then
               targ_fraction = targ_fraction * &
                 EXP(-(abs(ycent(j)) - sim_targetRadius)**2.0 / sigma2)
             endif
             if ( abs(zcent(k)) > sim_targetRadius ) then
               targ_fraction = targ_fraction * &
                 EXP(-(abs(zcent(k)) - sim_targetRadius)**2.0 / sigma2)
             endif
               !if ( xcent(i) <= 0 .and. &
               !     xcent(i) >= -sim_targetThickness .and. &
               !     abs(ycent(j)) <= sim_targetRadius .and. &
               !     abs(zcent(k)) <= sim_targetRadius) then
               !   species = TARG_SPEC
               !end if                 
           else
               r2 = (xcent(i)**2+ycent(j)**2+zcent(k)**2)
               if (r2 > sim_targetRadius**2) then
                  targ_fraction = EXP(-(sqrt(r2) - sim_targetRadius)**2 / sigma2)
              end if
           end if
           
           ! number fraction
           cham_fraction = sim_smallX + ( 1 - targ_fraction ) * ( 1 - 2 * sim_smallX )
           targ_fraction = sim_smallX + (     targ_fraction ) * ( 1 - 2 * sim_smallX )

           !cham_fraction = 0.01
           !targ_fraction = 0.99

            rho = cham_fraction *  sim_rhoCham + targ_fraction *  sim_rhoTarg
           tele = cham_fraction * sim_teleCham + targ_fraction * sim_teleTarg
           tion = cham_fraction * sim_tionCham + targ_fraction * sim_tionTarg
           trad = cham_fraction * sim_tradCham + targ_fraction * sim_tradTarg


           call Grid_putPointData(blockId, CENTER, DENS_VAR, EXTERIOR, axis, rho)
           call Grid_putPointData(blockId, CENTER, TEMP_VAR, EXTERIOR, axis, tele)

#ifdef FLASH_3T
           call Grid_putPointData(blockId, CENTER, TION_VAR, EXTERIOR, axis, tion)
           call Grid_putPointData(blockId, CENTER, TELE_VAR, EXTERIOR, axis, tele)

           ! Set up radiation energy density:
           call RadTrans_mgdEFromT(blockId, axis, trad, tradActual)
           call Grid_putPointData(blockId, CENTER, TRAD_VAR, EXTERIOR, axis, tradActual)
#endif
           call Grid_putPointData(blockID, CENTER, TARG_SPEC, EXTERIOR, axis, targ_fraction * sim_rhoTarg / rho)
           call Grid_putPointData(blockID, CENTER, CHAM_SPEC, EXTERIOR, axis, cham_fraction * sim_rhoCham / rho)

#ifdef BDRY_VAR
           call Grid_putPointData(blockId, CENTER, BDRY_VAR, EXTERIOR, axis, -1.0)
#endif


#if NFACE_VARS > 0
           !! In this case we initialized Az using the cell-cornered coordinates.
           if (sim_killdivb) then
              if (NDIM == 2) then
                 facexData(MAG_FACE_VAR,i,j,k)= 0.0
                 faceyData(MAG_FACE_VAR,i,j,k)= 0.0
                 if (NDIM>2) facezData(MAG_FACE_VAR,i,j,k)= 0.0
              endif
           endif
#endif
        enddo
     enddo
  enddo
#if NFACE_VARS > 0
  if (sim_killdivb) then
     call Grid_releaseBlkPtr(blockID,facexData,FACEX)
     call Grid_releaseBlkPtr(blockID,faceyData,FACEY)
     if (NDIM>2) call Grid_releaseBlkPtr(blockID,facezData,FACEZ)
  endif
#endif
  deallocate(xcent)
  deallocate(ycent)
  deallocate(zcent)

  return

end subroutine Simulation_initBlock
