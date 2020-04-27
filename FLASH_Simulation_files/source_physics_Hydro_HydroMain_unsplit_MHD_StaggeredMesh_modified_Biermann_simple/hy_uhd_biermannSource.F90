!!****if* source/physics/Hydro/HydroMain/unsplit/MHD_StaggeredMesh/hy_uhd_biermannSource
!!
!! NAME
!!
!!  hy_uhd_biermannSource
!!
!! SYNOPSIS
!!
!!  call hy_uhd_biermannSource( integer (IN) :: blockCount,
!!                         integer (IN) :: blockList(blockCount),
!!                         real    (IN) :: dt )
!!
!! DESCRIPTION
!!
!! Implement Biermann Battery Term as a source to the magnetic field.
!!
!! ARGUMENTS
!!
!!  blockCount -  the number of blocks in blockList
!!  blockList  -  array holding local IDs of blocks on which to advance
!!  dt         -  timestep
!!***

Subroutine hy_uhd_biermannSource ( blockCount, blockList, dt )
  use Grid_interface, ONLY: Grid_getBlkIndexLimits, &
                            Grid_getDeltas,         &
                            Grid_getBlkPtr,         &
                            Grid_releaseBlkPtr

  use Hydro_data, ONLY : hy_biermannCoef,  &
                         hy_useBiermann,   &
                         hy_useBiermann1T, &
                         hy_bier1TZ,       &
                         hy_bier1TA,       &
                         hy_avogadro,      &
                         hy_qele,          &
                         hy_speedOfLight,  &
                         hy_geometry,      &
                         hy_useBiermann3T, &
                         hy_eosModeAfter

  use hy_uhd_slopeLimiters, ONLY : minmod

  use Eos_interface, ONLY : Eos_wrapped, Eos_getAbarZbar
  
  implicit none

#include "Flash.h"
#include "constants.h"
#include "UHD.h"

  ! Arguments:
  integer, intent(IN) :: blockCount
  integer, intent(IN) :: blockList(blockCount)
  real,    intent(IN) :: dt

  ! Local Variables:
  real :: dens
  real :: abar
  real :: zbar
  real :: gradDensX, gradDensY
  real :: gradPresX, gradPresY
  real :: del(MDIM)
  real :: CL
  real :: QELE
  real :: NA
  real :: source
  real :: esource
  real :: ye, sumy
  real :: nele, nele_left, nele_rght, gradNelex, gradPelex, gradNeley, gradPeley
  real :: gradNelez, gradPelez!! FOR 3D
  real :: oldEmag, newEmag

  integer :: blockID
  integer :: blkLimits(LOW:HIGH,MDIM)
  integer :: blkLimitsGC(LOW:HIGH,MDIM)
  integer :: i, j, k, n

  real, pointer :: U(:,:,:,:)
  real(8), pointer, dimension(:,:,:,:) :: Bx,By,Bz,E

  if (.not. hy_useBiermann) return

  if (hy_useBiermann3T) return

  ! Begin:

  QELE = hy_qele
  CL = hy_speedOfLight
  NA = hy_avogadro

#ifdef FLASH_UHD_3T

  do n = 1, blockCount
     blockID = blockList(n)

     call Grid_getBlkIndexLimits(blockID, blkLimits, blkLimitsGC)
     call Grid_getBlkPtr(blockID, U, CENTER)
     call Grid_getDeltas(blockID, del)
     if(NDIM == 3) then
       call Grid_getBlkPtr(blockID,Bx,FACEX)
       call Grid_getBlkPtr(blockID,By,FACEY)
       call Grid_getBlkPtr(blockID,Bz,FACEZ)
       call Grid_getBlkPtr(blockID, E,SCRATCH)
       E(EX_SCRATCH_GRID_VAR, :,:,:) = 0
       E(EY_SCRATCH_GRID_VAR, :,:,:) = 0
       E(EZ_SCRATCH_GRID_VAR, :,:,:) = 0
     endif
     
     call Eos_wrapped(hy_eosModeAfter, blkLimits, blockID)


     do k = blkLimits(LOW, KAXIS)-1, blkLimits(HIGH,KAXIS)+1
        do j = blkLimits(LOW,JAXIS)-1, blkLimits(HIGH,JAXIS)+1
           do i = blkLimits(LOW,IAXIS)-1, blkLimits(HIGH,IAXIS)+1
! #ifdef FLASH_USM_MHD
#ifdef SHOK_VAR      
            if(U(SHOK_VAR,i,j,k) < 0.5) then
#endif  

            if (hy_geometry == CARTESIAN) then

              if (NDIM >= 2) then

                 ! Nele at i,j,k
                 call Eos_getAbarZbar(U(:,i,j,k), Ye=ye)
                 nele = ye * hy_avogadro * U(DENS_VAR,i,j,k)


                 !! dxNe, dyPe
                 ! Nele at i-1,j,k
                 call Eos_getAbarZbar(U(:,i-1,j,k), Ye=ye)
                 nele_left = ye * hy_avogadro * U(DENS_VAR,i-1,j,k)

                 ! Nele at i+1,j,k
                 call Eos_getAbarZbar(U(:,i+1,j,k), Ye=ye)
                 nele_rght = ye * hy_avogadro * U(DENS_VAR,i+1,j,k)

                 gradNelex=minmod(nele_rght-nele,nele-nele_left)/del(DIR_X)
                 gradPeley=minmod(U(PELE_VAR,i,j+1,k)-U(PELE_VAR,i,j,  k),&
                                  U(PELE_VAR,i,j,  k)-U(PELE_VAR,i,j-1,k))/del(DIR_Y)

                 !! dxPe, dyNe
                 ! Nele at i,j-1,k
                 call Eos_getAbarZbar(U(:,i,j-1,k), Ye=ye)
                 nele_left = ye * hy_avogadro * U(DENS_VAR,i,j-1,k)
           
                 ! Nele at i,j+1,k
                 call Eos_getAbarZbar(U(:,i,j+1,k), Ye=ye)
                 nele_rght = ye * hy_avogadro * U(DENS_VAR,i,j+1,k)

                 gradNeley=minmod(nele_rght-nele,nele-nele_left)/del(DIR_Y)
                 gradPelex=minmod(U(PELE_VAR,i+1,j,k)-U(PELE_VAR,i,  j,k),&
                                  U(PELE_VAR,i,  j,k)-U(PELE_VAR,i-1,j,k))/del(DIR_X)

                 ! Store old magnetic energy
                 oldEmag = 0.5*(U(MAGX_VAR,i,j,k)**2.0 + U(MAGY_VAR,i,j,k)**2.0 + U(MAGZ_VAR,i,j,k)**2.0 )

                 ! Add Battery effect to Bz
                 source = dt*(gradPelex*gradNeley - gradPeley*gradNelex)/(hy_qele*nele**2)

                 U(MAGZ_VAR,i,j,k) = U(MAGZ_VAR,i,j,k) + source
                 Bz(MAG_FACE_VAR, i,j,k  ) = Bz(MAG_FACE_VAR, i,j,k  ) + source/2.0
                 Bz(MAG_FACE_VAR, i,j,k+1) = Bz(MAG_FACE_VAR, i,j,k+1) + source/2.0

              endif

              if (NDIM ==3) then

                 !! dxNe, dyPe
                 ! Nele at i,j,k-1
                 call Eos_getAbarZbar(U(:,i,j,k-1), Ye=ye)
                 nele_left = ye * hy_avogadro * U(DENS_VAR,i,j,k-1)

                 ! Nele at i,j,k+1
                 call Eos_getAbarZbar(U(:,i,j,k+1), Ye=ye)
                 nele_rght = ye * hy_avogadro * U(DENS_VAR,i,j,k+1)                

                 gradNelez=minmod(nele_rght-nele,nele-nele_left)/del(DIR_Z)
                 gradPelez=minmod(U(PELE_VAR,i,j,k+1)-U(PELE_VAR,i,j,  k),&
                                  U(PELE_VAR,i,j,  k)-U(PELE_VAR,i,j,k-1))/del(DIR_Z)

                 ! Add Battery effect to Bx and By.
                 source = dt*(gradPeley*gradNelez - gradPelez*gradNeley)/(hy_qele*nele**2)
                 U(MAGX_VAR,i,j,k) = U(MAGX_VAR,i,j,k) + source 
                 Bx(MAG_FACE_VAR, i  ,j,k) = Bx(MAG_FACE_VAR, i  ,j,k) + source/2.0
                 Bx(MAG_FACE_VAR, i+1,j,k) = Bx(MAG_FACE_VAR, i+1,j,k) + source/2.0

                 source = dt*(gradPelez*gradNelex - gradPelex*gradNelez)/(hy_qele*nele**2)
                 U(MAGY_VAR,i,j,k) = U(MAGY_VAR,i,j,k) + source
                 By(MAG_FACE_VAR, i,j  ,k) = By(MAG_FACE_VAR, i,j  ,k) + source/2.0
                 By(MAG_FACE_VAR, i,j+1,k) = By(MAG_FACE_VAR, i,j+1,k) + source/2.0

              end if

              ! New magnetic energy
              newEmag = 0.5*(U(MAGX_VAR,i,j,k)**2.0 + U(MAGY_VAR,i,j,k)**2.0 + U(MAGZ_VAR,i,j,k)**2.0 )
              esource = (newEmag - oldEmag) / U(DENS_VAR,i,j,k)

              U(EINT_VAR,i,j,k) = U(EINT_VAR,i,j,k) - esource
              U(EELE_VAR,i,j,k) = U(EELE_VAR,i,j,k) - esource
              U(ENER_VAR,i,j,k) = U(ENER_VAR,i,j,k) - esource


! #endif

            end if 
                !! end if CARTESIAN

            if (hy_geometry == CYLINDRICAL) then

              if (NDIM == 2) then

                 ! Nele at i,j,k
                 call Eos_getAbarZbar(U(:,i,j,k), Ye=ye)
                 nele = ye * hy_avogadro * U(DENS_VAR,i,j,k)

                 !! dxNe, dyPe
                 ! Nele at i-1,j,k
                 call Eos_getAbarZbar(U(:,i-1,j,k), Ye=ye)
                 nele_left = ye * hy_avogadro * U(DENS_VAR,i-1,j,k)

                 ! Nele at i+1,j,k
                 call Eos_getAbarZbar(U(:,i+1,j,k), Ye=ye)
                 nele_rght = ye * hy_avogadro * U(DENS_VAR,i+1,j,k)

                 gradNelex=minmod(nele_rght-nele,nele-nele_left)/del(DIR_X)
                 gradPeley=minmod(U(PELE_VAR,i,j+1,k)-U(PELE_VAR,i,j,  k),&
                                  U(PELE_VAR,i,j,  k)-U(PELE_VAR,i,j-1,k))/del(DIR_Y)

                 !! dxPe, dyNe
                 ! Nele at i,j-1,k
                 call Eos_getAbarZbar(U(:,i,j-1,k), Ye=ye)
                 nele_left = ye * hy_avogadro * U(DENS_VAR,i,j-1,k)
           
                 ! Nele at i,j+1,k
                 call Eos_getAbarZbar(U(:,i,j+1,k), Ye=ye)
                 nele_rght = ye * hy_avogadro * U(DENS_VAR,i,j+1,k)

                 gradNeley=minmod(nele_rght-nele,nele-nele_left)/del(DIR_Y)
                 gradPelex=minmod(U(PELE_VAR,i+1,j,k)-U(PELE_VAR,i,  j,k),&
                                  U(PELE_VAR,i,  j,k)-U(PELE_VAR,i-1,j,k))/del(DIR_X)

                 ! Store old magnetic energy
                 oldEmag = 0.5*U(MAGZ_VAR,i,j,k)**2.0

                 ! Add Battery effect to Bz (which, in Cylindrical, corresponds to Bphi)
                 source = dt*(gradPeley*gradNelex - gradPelex*gradNeley)/(hy_qele*nele**2)
                 U(MAGZ_VAR,i,j,k) = U(MAGZ_VAR,i,j,k) + source

                 ! New magnetic energy
                 newEmag = 0.5*U(MAGZ_VAR,i,j,k)**2.0
                 esource = (newEmag - oldEmag) / U(DENS_VAR,i,j,k)

                 U(EINT_VAR,i,j,k) = U(EINT_VAR,i,j,k) - esource
                 U(EELE_VAR,i,j,k) = U(EELE_VAR,i,j,k) - esource
                 U(ENER_VAR,i,j,k) = U(ENER_VAR,i,j,k) - esource

               end if

            end if

#ifdef SHOK_VAR   
           endif
#endif

           end do
        end do
     end do


     call Grid_releaseBlkPtr(blockID,U,CENTER)
     if(NDIM == 3) then
       E(EX_SCRATCH_GRID_VAR, :,:,:) = 0
       E(EY_SCRATCH_GRID_VAR, :,:,:) = 0
       E(EZ_SCRATCH_GRID_VAR, :,:,:) = 0
       call Grid_releaseBlkPtr(blockID,Bx,FACEX)
       call Grid_releaseBlkPtr(blockID,By,FACEY)
       call Grid_releaseBlkPtr(blockID,Bz,FACEZ)
       call Grid_releaseBlkPtr(blockID, E,SCRATCH)
     endif
     call Eos_wrapped(hy_eosModeAfter, blkLimits, blockID)
  end do

#endif 
  ! end of #ifdef FLASH_UHD_3T

  return

End Subroutine hy_uhd_biermannSource
