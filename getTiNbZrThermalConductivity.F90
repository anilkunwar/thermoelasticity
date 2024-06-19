    !-----------------------------------------------------
    ! Written By: Anil Kunwar (Original 2015-03-13) (Modification 2021-11-16)
    ! material property user defined function for ELMER:
    ! Thermal conductivity of Ti0.8455Nb0.0765Zr0.0779 fitted as a function of temperature
    ! (kth_tinbzr)solid = As*(T)^2 + Bs*(T) + Cs, where A = -9.12e-07 W K/m^3 and B = 4.34e-03 W/m K^2 and C = 18.0 W/mK (298.0 K < T < 1973.0 K)
    ! X.P. Zhang  et al. J. Mater. Sci. (2005), Vol. 40:4911-4916
    ! https://pubs.aip.org/avs/jvb/article/29/6/061803/104835/Laser-nitriding-of-niobium-for-application-to
    ! https://www.sciencedirect.com/science/article/pii/0022311595001107
    ! https://periodictable.com/Elements/022/data.html (Value of C, E)
    ! https://periodictable.com/Elements/040/data.html (Value of C, E)
    ! https://periodictable.com/Elements/041/data.html (Value of C, E)
    ! https://link.springer.com/content/pdf/10.1007/s10853-005-0418-0.pdf
    ! (kth_tinbzr)liquid = Al*(T)^2 + Bl*T + Cl, where Al = 1.95E-07 W/m K^3 ,Bl = 1.55E-02 W/m K^2 and Cl = 5.73 W/m K (1973.0 K < T < 2500 K)
    ! https://www.sciencedirect.com/science/article/pii/S0167732220373803
    !-----------------------------------------------------
    FUNCTION getThermalConductivity( model, n, temp ) RESULT(thcondt)
    ! modules needed
    USE DefUtils
    IMPLICIT None
    ! variables in function header
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: temp, thcondt

    ! variables needed inside function
    REAL(KIND=dp) :: alphas, betas, deltas, &
    alphal, betal, deltal, refTemp
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    ! get pointer on list for material
    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getThermalConductivity', 'No material found')
    END IF

    ! read in reference conductivity at reference temperature
    !refThCond = GetConstReal( material, 'Reference Thermal Conductivity C Solid TiNbZr',GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getThermalConductivity', 'Reference Thermal Conductivity Solid TiNbZr not found')
    !END IF

    ! read in Temperature Coefficient of Resistance
    alphas = GetConstReal( material, 'Cond Coeff A Solid TiNbZr', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'slope of thermal conductivity-temperature curve solid not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    betas = GetConstReal( material, 'Cond Coeff B Solid TiNbZr', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Coefficientt of T2 term solid Ti not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    deltas = GetConstReal( material, 'Cond Coeff C Solid TiNbZr', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Coefficientt of T2 term solid Ti not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    alphal = GetConstReal( material, 'Cond Coeff A Liquid TiNbZr', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'slope of thermal conductivity-temperature curve solid not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    betal = GetConstReal( material, 'Cond Coeff B Liquid TiNbZr', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Coefficientt of T2 term solid Ti not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    deltal = GetConstReal( material, 'Cond Coeff C Liquid TiNbZr', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Coefficientt of T2 term solid Ti not found')
    END IF

    ! read in reference temperature
    refTemp = GetConstReal( material, 'Melting Point Temperature TiNbZr', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Reference Temperature not found')
    END IF


    ! compute density conductivity
    IF (refTemp <= temp) THEN ! check for physical reasonable temperature
       CALL Warn('getThermalConductivity', 'The Ti material is in liquid state.')
            !CALL Warn('getThermalConductivity', 'Using density reference value')
    !thcondt = 1.11*(refThCond + alpha*(temp))
    thcondt = deltal + betal*(temp) + alphal*(temp**2) 
    ELSE
    thcondt = deltas + betas*(temp) + alphas*(temp**2) 
    END IF

    END FUNCTION getThermalConductivity

