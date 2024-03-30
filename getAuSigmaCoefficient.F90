    !-----------------------------------------------------
    ! Written By: Anil Kunwar (Original 2015-03-13) (Modification 2024-03-31)
    ! material property user defined function for ELMER:
    ! Coefficient of thermal stress as a function of temperature
    ! sigma_T = E*coefficient*(T-T_ref)
    ! coeffcient = switch*alpha
    ! where alpha_i = a_0i + a_1i*T
    ! The switch is 0 for liquid phase and 1 for solid phase
    ! References
    !-----------------------------------------------------
    FUNCTION getThermalStressCoefficient( model, n, temp ) RESULT(thsigmacoeff)
    ! modules needed
    USE DefUtils
    IMPLICIT None
    ! variables in function header
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: temp, thsigmacoeff, tscaler

    ! variables needed inside function
    REAL(KIND=dp) :: refSolThExp, refLiqThExp,refTemp, &
    alphas, alphal 
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    ! get pointer on list for material
    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getThermalStressCoefficient', 'No material found')
    END IF

    ! read in reference conductivity at reference temperature
    refSolThExp = GetConstReal( material, 'Reference Thermal Expansivity of Solid Au',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalStressCoefficient', 'Reference Thermal Expansivity of Solid Au not found')
    END IF

    ! read in Temperature Coefficient of Resistance
    alphas = GetConstReal( material, 'Exp Coeff Solid TiNbZr', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalStressCoefficient', 'slope of thermal expansivity-temperature curve solid not found')
    END IF
    
    ! read in reference conductivity at reference temperature
    refLiqThExp = GetConstReal( material, 'Reference Thermal Expansivity of Liquid Au',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalStressCoefficient', 'Reference Thermal Expansivity of Liquid Au not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    alphal = GetConstReal( material, 'Exp Coeff Liquid Au', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalStressCoefficient', 'slope of thermal expansivity-temperature curve Liquid Au not found')
    END IF
    
    
    ! read in reference temperature
    refTemp = GetConstReal( material, 'Melting Point Temperature Au', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalStressCoefficient', 'Reference Temperature not found')
    END IF
    
    ! read in the temperature scaling factor
    tscaler = GetConstReal( material, 'Tscaler', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalStressCoefficient', 'Scaling Factor for T not found')
    END IF


    ! compute density conductivity
    IF (refTemp <= temp) THEN ! check for physical reasonable temperature
       CALL Warn('getThermalStressCoefficient', 'The Ti material is in liquid state.')
            !CALL Warn('getThermalStressCoefficient', 'Using density reference value')
    !thsigmacoeff = 0*(refLiqThExp + alphal*((tscaler)*temp))
    thsigmacoeff = 0 ! whatever the value of refLiqThExp and alphal, the switch will render the coefficient zero
    ELSE
    thsigmacoeff = refSolThExp + alphas*((tscaler)*temp) ! the switch is 1 here
    END IF

    END FUNCTION getThermalStressCoefficient
