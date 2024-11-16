    !-----------------------------------------------------
    ! material property user defined function for ELMER:
    ! Elastic modulus of solid alloy as piecewise function of vonMises stress
    ! yield strength (sigma_yield) is obtained from experiments
    ! When sigma <= sigma_yield , E = E_0 = 68.0 GPa
    ! When sigma > sigma_yield , E = simga/((sigma/K_H))**(1/n)
    ! where, strength coefficient K_H = 381.08 MPa and strain hardening coefficient = n = 0.103; m = 1/n = 9.7087
    ! Reference: Natesan et al., Materials 2019.
    ! https://www.mdpi.com/1996-1944/12/18/3033
    ! Reference: https://github.com/anilkunwar/elmerfem/blob/devel/fem/tests/AnisotropicThermalConductivity/GetThermalConductivityTensor.F90
    !-----------------------------------------------------
    FUNCTION getPlasticityTensor( model, n, stress ) RESULT(elast)
    ! modules needed
    USE DefUtils
    IMPLICIT None
    ! variables in function header
    TYPE(Model_t) :: model
    INTEGER :: n
    !REAL(KIND=dp) :: stress, elast
    REAL(KIND=dp),POINTER ::  stress(:,:) 
    REAL(KIND=dp),POINTER ::  elast(:,:) ! this size needs to be consistent with the sif file!


    ! variables needed inside function
    !REAL(KIND=dp) :: refElast, yieldsigma,   &
    !strengthcoeff, mcoeff, pfactor
    REAL(KIND=dp),POINTER ::  refElast(:,:), yieldsigma(:,:)
    REAL(KIND=dp) :: strengthcoeff, mcoeff,  &
    pfactor
    !REAL(KIND=dp) :: refElast, yieldsigma,   &
    !strengthcoeff, mcoeff, pfactor
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    ! get pointer on list for material
    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getPlasticityTensor', 'No material found')
    END IF

    ! read in reference conductivity at reference temperature
    ! Now the coefficient refElast(i,j) are the function of E_x, E_y, and v_ij
    !refElast = GetConstReal( material, 'isotropic modulus in elastic regime in Pa',GotIt)
    refElast(1,1) = GetConstReal( material, 'refElast(1,1) in elastic regime in Pa',GotIt)
    refElast(1,2) = GetConstReal( material, 'refElast(1,2) in elastic regime in Pa',GotIt)
    refElast(1,3) = GetConstReal( material, 'refElast(1,3) in elastic regime in Pa',GotIt)
    refElast(2,1) = GetConstReal( material, 'refElast(2,1) in elastic regime in Pa',GotIt)
    refElast(2,2) = GetConstReal( material, 'refElast(2,2) in elastic regime in Pa',GotIt)
    refElast(2,3) = GetConstReal( material, 'refElast(2,3) in elastic regime in Pa',GotIt)
    refElast(3,1) = GetConstReal( material, 'refElast(3,1) in elastic regime in Pa',GotIt)
    refElast(3,2) = GetConstReal( material, 'refElast(3,2) in elastic regime in Pa',GotIt)
    refElast(3,3) = GetConstReal( material, 'refElast(3,3) in elastic regime in Pa',GotIt)
    !refDenst = GetConstReal( material, 'Solid_ti_rho_constant',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticity', 'Constant Youngs modulus not found')
    END IF

    ! read in Temperature Coefficient of Resistance
    !yieldsigma = GetConstReal( material, 'Yield strength of the alloy materials in Pa', GotIt)
    yieldsigma(1,1) = GetConstReal( material, 'yieldsigma(1,1) of the alloy materials in Pa',GotIt)
    yieldsigma(1,2) = GetConstReal( material, 'yieldsigma(1,2) of the alloy materials in Pa',GotIt)
    yieldsigma(1,3) = GetConstReal( material, 'yieldsigma(1,3) of the alloy materials in Pa',GotIt)
    yieldsigma(2,1) = GetConstReal( material, 'yieldsigma(2,1) of the alloy materials in Pa',GotIt)
    yieldsigma(2,2) = GetConstReal( material, 'yieldsigma(2,2) of the alloy materials in Pa',GotIt)
    yieldsigma(2,3) = GetConstReal( material, 'yieldsigma(2,3) of the alloy materials in Pa',GotIt)
    yieldsigma(3,1) = GetConstReal( material, 'yieldsigma(3,1) of the alloy materials in Pa',GotIt)
    yieldsigma(3,2) = GetConstReal( material, 'yieldsigma(3,2) of the alloy materials in Pa',GotIt)
    yieldsigma(3,3) = GetConstReal( material, 'yieldsigma(3,3) of the alloy materials in Pa',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticityTensor', 'Sigma yield not found')
    END IF
     
    ! read in pseudo reference conductivity at reference temperature of liquid
    strengthcoeff = GetConstReal( material, 'Strength coefficient in Ramberg-Osgood equation',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticityTensor', 'Density Coefficient Al of Liquid AlMgSiZr not found')
    END IF

    ! read in reference temperature
    mcoeff = GetConstReal( material, 'Reciprocal of strain hardening coefficient', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticityTensor', '1/n factor not found')
    END IF
    
    ! read in reference temperature
    pfactor = GetConstReal( material, 'stabilizer for the plasticity simulation', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticityTensor', 'stabilizer factor not found')
    END IF
    
    ! read in the temperature scaling factor
    !tscaler = GetConstReal( material, 'Tscaler', GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getThermalConductivity', 'Scaling Factor for T not found')
    !END IF

    ! compute density conductivity
    !IF (yieldsigma <= stress) THEN ! check for physical reasonable temperature
    !   CALL Warn('getPlasticity', 'The AlMgSiZr material undergoing plastic deformation.')
    !        !CALL Warn('getPlasticity', 'Using density reference value')
    !!denst = 1.11*(refDenst + alpha*(temp))
    !elast =  pfactor*stress/(stress/strengthcoeff)**mcoeff 
    !ELSE
    !elast =  refElast
    !END IF
    
    IF (yieldsigma(1,1) <= stress(1,1)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(1,1) =  pfactor*stress(1,1)/(stress(1,1)/strengthcoeff)**mcoeff 
    ELSE
    elast(1,1) =  refElast(1,1)
    END IF
    
    IF (yieldsigma(1,2) <= stress(1,2)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(1,2) =  pfactor*stress(1,2)/(stress(1,2)/strengthcoeff)**mcoeff 
    ELSE
    elast(1,2) =  refElast(1,2)
    END IF
    
    IF (yieldsigma(1,3) <= stress(1,3)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(1,3) =  pfactor*stress(1,3)/(stress(1,3)/strengthcoeff)**mcoeff 
    ELSE
    elast(1,3) =  refElast(1,3)
    END IF
    
    IF (yieldsigma(2,1) <= stress(2,1)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(2,1) =  pfactor*stress(2,1)/(stress(2,1)/strengthcoeff)**mcoeff 
    ELSE
    elast(2,1) =  refElast(2,1)
    END IF
    
    IF (yieldsigma(2,2) <= stress(2,2)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(2,2) =  pfactor*stress(2,2)/(stress(2,2)/strengthcoeff)**mcoeff 
    ELSE
    elast(2,2) =  refElast(2,2)
    END IF
    
    IF (yieldsigma(2,3) <= stress(2,3)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(2,3) =  pfactor*stress(2,3)/(stress(2,3)/strengthcoeff)**mcoeff 
    ELSE
    elast(2,3) =  refElast(2,3)
    END IF
    
    IF (yieldsigma(3,1) <= stress(3,1)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(3,1) =  pfactor*stress(3,1)/(stress(3,1)/strengthcoeff)**mcoeff 
    ELSE
    elast(3,1) =  refElast(3,1)
    END IF
    
    IF (yieldsigma(3,2) <= stress(3,2)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(3,2) =  pfactor*stress(3,2)/(stress(3,2)/strengthcoeff)**mcoeff 
    ELSE
    elast(3,2) =  refElast(3,2)
    END IF
    
    IF (yieldsigma(3,3) <= stress(3,3)) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticityTensor', 'The Au or TiNbZr material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast(3,3) =  pfactor*stress(3,3)/(stress(3,3)/strengthcoeff)**mcoeff 
    ELSE
    elast(3,3) =  refElast(3,3)
    END IF

    END FUNCTION getPlasticityTensor

