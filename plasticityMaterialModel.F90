    !-----------------------------------------------------
    ! material property user defined function for ELMER:
    ! Elastic modulus of solid alloy as piecewise function of vonMises stress
    ! yield strength (sigma_yield) is obtained from experiments
    ! When sigma <= sigma_yield , E = E_0 = 68.0 GPa
    ! When sigma > sigma_yield , E = simga/((sigma/K_H))**(1/n)
    ! where, strength coefficient K_H = 381.08 MPa and strain hardening coefficient = n = 0.103; m = 1/n = 9.7087
    ! Reference: Natesan et al., Materials 2019.
    ! https://www.mdpi.com/1996-1944/12/18/3033
    !-----------------------------------------------------
    FUNCTION getElasticity( model, n, stress ) RESULT(elast)
    ! modules needed
    USE DefUtils
    IMPLICIT None
    ! variables in function header
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: stress, elast

    ! variables needed inside function
    REAL(KIND=dp) :: refElast, yieldsigma   &
    strengthcoeff, mcoeff
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    ! get pointer on list for material
    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getElasticity', 'No material found')
    END IF

    ! read in reference conductivity at reference temperature
    refElast = GetConstReal( material, 'Isotropic elastic modulus in elastic regime in Pa',GotIt)
    !refDenst = GetConstReal( material, 'Solid_ti_rho_constant',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getElasticity', 'Constant Youngs modulus not found')
    END IF

    ! read in Temperature Coefficient of Resistance
    yieldsigma = GetConstReal( material, 'Yield strength of the alloy materials in Pa', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getElasticity', 'Sigma yield not found')
    END IF
     
    ! read in pseudo reference conductivity at reference temperature of liquid
    strengthcoeff = GetConstReal( material, 'Strength coefficient in Ramberg-Osgood equation',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getElasticity', 'Density Coefficient Al of Liquid AlMgSiZr not found')
    END IF

    ! read in reference temperature
    mcoeff = GetConstReal( material, 'Reciprocal of strain hardening coefficient', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getElasticity', '1/n factor not found')
    END IF
    
    ! read in the temperature scaling factor
    !tscaler = GetConstReal( material, 'Tscaler', GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getThermalConductivity', 'Scaling Factor for T not found')
    !END IF

    ! compute density conductivity
    IF (yieldsigma <= stress) THEN ! check for physical reasonable temperature
       CALL Warn('getElasticity', 'The AlMgSiZr material undergoing plastic deformation.')
            !CALL Warn('getElasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast =  stress/(stress/strengthcoeff)**mcoeff 
    ELSE
    elast =  refElast
    END IF

    END FUNCTION getElasticity

