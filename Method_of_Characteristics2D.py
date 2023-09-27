#TOBERA DE LAVAL 2D POR EL METODO DE LAS CARACTERISTICAS


# Resolucion del problema cuasi-unidimensional.
P1 = P1Result #Decision del punto de operacion.
gamma = 1.2529712919314542 
R = 349.52459961214936 
P2  =   P3
v2  =   np.sqrt(((2*gamma)/(gamma-1))*R*T1*((P1/P3)**((gamma-1)/gamma)-1))
G   =   regressionRate(P1,coefs[0],coefs[1])*Aref*rho#(Tf*9.81)/v2
T   =   (G*v2/9.81)
At  =   G/(P1*gamma*np.sqrt((2/(gamma+1))**((gamma+1)/(gamma-1))/(gamma*R*T1)))
M2  =   np.sqrt((2/(gamma-1))*((P1/P2)**((gamma-1)/gamma)-1))
e   =   (1/M2)*((2/(gamma+1))*(1+((gamma-1)/2)*M2**2))**((gamma+1)/(2*(gamma-1)))
A2  =   At*e
Dt  =   2*np.sqrt(At/np.pi)
D2  =   2*np.sqrt(A2/np.pi)
Cf  =   (T*9.81)/(At*P1)
cStar = P1*At/G
Isp = cStar*Cf/9.81
T2 = T1/(1+((gamma-1)/2)*M2**2)
Tt = T1/((gamma+1)/2)
Pr = P1/((gamma+1)/2)**(gamma/(gamma-1))
P2 = P1/(1+((gamma-1)/2)*M2**2)**(gamma/(gamma-1))


# Definicion de la funcion que ejecuta el metodo de las caracteristicas.
def MOC_BELL(mach_Outlet,n_characteristics,throat_diameter,gamma_parameter,throat_Factor):

    resultCols = ["X","Y","Theta","Nu","M","Mu","C+(dy/dx)","C-(dy/dx)","K+","K-"]
    prandtlMayerCols = ["Mach","Nu(M)"]

    Me = mach_Outlet
    n = n_characteristics
    Dt = throat_diameter
    gamma = gamma_parameter
    
    beta = throat_Factor*(Dt/2)

    gK = (gamma+1)/(gamma-1)
    Mmax = 10

    # Funcion para calcular el angulo de expansion de Prandtl-Meyer.
    def nu(M):
        return (np.sqrt(gK)*math.atan(np.sqrt((1/gK)*(M**2-1))) - math.atan(np.sqrt(M**2-1)))*(180/np.pi)

    prandtlMayerCols = ["Mach","Nu(M)"]
    prandtlMayerArr = []

    for M in np.arange(1,Mmax,0.0001):
        prandtlMayerArr.append([M,nu(M)])
    prandtlMayerArr.append([Mmax,nu(Mmax)])

    prandtlMayerData = pd.DataFrame(prandtlMayerArr,columns=prandtlMayerCols)

    # Funcion para calcular el inverso del angulo de expansion de Prandtl-Meyer.
    def inverse_nu(_nu_):

        filteredValues = prandtlMayerData[prandtlMayerData['Nu(M)'].between(_nu_ -0.01 , _nu_ +0.01 )]

        for i in range(1,len(filteredValues)-1):
            
            nuLow = filteredValues["Nu(M)"].to_numpy()[i-1]
            nuUp = filteredValues["Nu(M)"].to_numpy()[i+1]

            machLow = filteredValues["Mach"].to_numpy()[i-1]
            machUp = filteredValues["Mach"].to_numpy()[i+1]

            if nuLow < _nu_ < nuUp:
                mach = ((machLow-machUp)/(nuLow-nuUp))*(_nu_-nuLow)+machLow
                break

            else: mach = 1

        return mach

    # Funcion para calcular el angulo del cono de sonido.
    def mu(M):
        return math.asin(1/M)*(180/np.pi)

    # Iniciacion del primer punto (0) en la garganta.
    p0i_num = []
    p0i_posX = np.zeros(n)
    p0i_posY = np.zeros(n)

    p0_nu = nu(Me)
    p0_thetaMax = p0_nu/2

    p0i_theta = np.zeros(n)
    p0i_nu = np.zeros(n)
    p0i_mach = np.zeros(n)
    p0i_mu = np.zeros(n)

    p0i_Cpls_m = np.zeros(n)
    p0i_Cmin_m = np.zeros(n)

    p0i_Kpls = np.zeros(n)
    p0i_Kmin = np.zeros(n)

    thetaStep = p0_thetaMax/n

    # Calculo de las variables en el tren de ondas que emana de la garganta.
    for i in range(0,n):
        
        p0i_num.append("0," + str(i+1))
        p0i_theta[i] = thetaStep*(1+i)
        p0i_nu[i] = p0i_theta[i]
        p0i_mach[i] = inverse_nu(p0i_nu[i])
        p0i_mu[i] = mu(p0i_mach[i])
        p0i_Kpls[i] = p0i_theta[i] - p0i_nu[i]
        p0i_Kmin[i] = p0i_theta[i] + p0i_nu[i]
        p0i_posX[i] = beta*np.sin(np.radians(p0i_theta[i]))
        p0i_posY[i] = Dt/2 + beta*(1 - np.cos(np.radians(p0i_theta[i])))

    p0i_results = np.transpose(np.vstack([
            p0i_posX,
            p0i_posY,
            p0i_theta,
            p0i_nu,
            p0i_mach,
            p0i_mu,
            p0i_Cpls_m,
            p0i_Cmin_m,
            p0i_Kpls,
            p0i_Kmin
            ]))
        
    p0i_results = pd.DataFrame(p0i_results,index=p0i_num,columns=resultCols).rename_axis('Points')
    dataArr = []
    tagArr = []

    for i in range(1,n+2):
        dataArr.append(np.empty((i,10), dtype='float'))
        tagArr.append(np.empty((i), dtype='U3'))
    del dataArr[0]
    del tagArr[0]

    dataArr.reverse()
    tagArr.reverse()

    ACount = 1
    WCount = 1
    ICount = 1

    # Se define una discretizacion especial de la region a calcular, donde cada vector de puntos j+1 tiene n-1 elementos.
    for i in range(0,len(tagArr)):
        for j in range(0,len(tagArr[i])):

            if j == 0:
                tag = "A" + str(ACount)
                ACount +=1

            elif j == len(tagArr[i])-1:
                tag = "W" + str(WCount)
                WCount +=1

            else:
                tag = "I" + str(ICount)
                ICount +=1

            tagArr[i][j] = tag

    for i in range(0,len(dataArr)):

        if i == 0: # Primer conjunto de puntos que parten de la garganta.
            for j in range(0,len(dataArr[i])):

                if j == 0: #Puntos en el eje.
                    pi_theta    = 0
                    pi_Kmin     = p0i_results["K-"][i]
                    pi_nu       = pi_Kmin - pi_theta
                    pi_Kplus    = pi_theta - pi_nu
                    pi_mach     = inverse_nu(pi_nu)
                    pi_mu       = mu(pi_mach)
                    pi_Cplus_m  = 0
                    pi_Cmin_m   = math.tan(np.radians(((pi_theta + p0i_results["Theta"][i]) - (pi_mu + p0i_results["Mu"][i]))/2))
                    pi_posY     = 0
                    pi_posX     = -p0i_results["Y"][i]/pi_Cmin_m
                    dataArr[i][j] = np.array([pi_posX,pi_posY,pi_theta,pi_nu,pi_mach,pi_mu,pi_Cplus_m,pi_Cmin_m,pi_Kplus,pi_Kmin])

                elif j == len(dataArr[i])-1: #Puntos en la pared.

                    pi_theta    = dataArr[i][j-1][2]
                    pi_nu       = dataArr[i][j-1][3]
                    pi_Kplus    = dataArr[i][j-1][8]
                    pi_Kmin     = pi_theta + pi_nu
                    pi_mach     = inverse_nu(pi_nu)
                    pi_mu       = mu(pi_mach)
                    pi_Cplus_m  = math.tan(np.radians(((pi_theta + dataArr[i][j-1][2]) + (pi_mu + dataArr[i][j-1][5]))/2))
                    pi_Cmin_m   = math.tan(np.radians((p0_thetaMax + pi_theta)/2))
                    pi_posY     = (dataArr[i][j-1][1]*pi_Cmin_m - p0i_results["Y"][i]*pi_Cplus_m + pi_Cplus_m*pi_Cmin_m*(p0i_results["X"][i] - dataArr[i][j-1][0]))/(pi_Cmin_m - pi_Cplus_m)
                    pi_posX     = (pi_Cplus_m*dataArr[i][j-1][0] - pi_Cmin_m*p0i_results["X"][i] + p0i_results["Y"][i] - dataArr[i][j-1][1])/(pi_Cplus_m - pi_Cmin_m)
                    dataArr[i][j] = np.array([pi_posX,pi_posY,pi_theta,pi_nu,pi_mach,pi_mu,pi_Cplus_m,pi_Cmin_m,pi_Kplus,pi_Kmin])

                else: #Puntos interiores.

                    pi_Kplus    = dataArr[i][j-1][8]
                    pi_Kmin     = p0i_results["K-"][i+j]
                    pi_theta    = (pi_Kmin + pi_Kplus)/2
                    pi_nu       = (pi_Kmin - pi_Kplus)/2
                    pi_mach     = inverse_nu(pi_nu)
                    pi_mu       = mu(pi_mach)
                    pi_Cplus_m  = math.tan(np.radians(((pi_theta + dataArr[i][j-1][2]) + (pi_mu + dataArr[i][j-1][5]))/2))
                    pi_Cmin_m   = math.tan(np.radians(((pi_theta + p0i_results["Theta"][i+j]) - (pi_mu + p0i_results["Mu"][i+j]))/2))
                    pi_posY     = (dataArr[i][j-1][1]*pi_Cmin_m - p0i_results["Y"][i+j]*pi_Cplus_m + pi_Cplus_m*pi_Cmin_m*(p0i_results["X"][i+j] - dataArr[i][j-1][0]))/(pi_Cmin_m - pi_Cplus_m)
                    pi_posX     = (pi_Cplus_m*dataArr[i][j-1][0] - pi_Cmin_m*p0i_results["X"][i+j] + p0i_results["Y"][i+j] - dataArr[i][j-1][1])/(pi_Cplus_m - pi_Cmin_m)
                    dataArr[i][j] = np.array([pi_posX,pi_posY,pi_theta,pi_nu,pi_mach,pi_mu,pi_Cplus_m,pi_Cmin_m,pi_Kplus,pi_Kmin])      


        if 0 < i < len(dataArr)-1: #Puntos intermedios entre la garganta y la salida.
            for j in range(0,len(dataArr[i])):

                if j == 0: #Puntos en el eje.
                    pi_theta    = 0
                    pi_Kmin     = dataArr[i-1][j+1][9]
                    pi_nu       = pi_Kmin
                    pi_Kplus    = pi_theta - pi_nu
                    pi_mach     = inverse_nu(pi_nu)
                    pi_mu       = mu(pi_mach)
                    pi_Cplus_m  = 0
                    pi_Cmin_m   = math.tan(np.radians(((pi_theta + dataArr[i-1][j+1][2]) - (pi_mu + dataArr[i-1][j+1][5]))/2))
                    pi_posY     = 0
                    pi_posX     = dataArr[i-1][j+1][0] - dataArr[i-1][j+1][1]/pi_Cmin_m
                    dataArr[i][j] = np.array([pi_posX,pi_posY,pi_theta,pi_nu,pi_mach,pi_mu,pi_Cplus_m,pi_Cmin_m,pi_Kplus,pi_Kmin])

                elif j == len(dataArr[i])-1: #Puntos en la pared.

                    pi_theta    = dataArr[i][j-1][2]
                    pi_nu       = dataArr[i][j-1][3]
                    pi_Kplus    = dataArr[i][j-1][8]
                    pi_Kmin     = pi_theta + pi_nu
                    pi_mach     = inverse_nu(pi_nu)
                    pi_mu       = mu(pi_mach)
                    pi_Cplus_m  = math.tan(np.radians(((pi_theta + dataArr[i][j-1][2]) + (pi_mu + dataArr[i][j-1][5]))/2))
                    pi_Cmin_m   = math.tan(np.radians(((pi_theta + dataArr[i-1][-1][2])/2)))
                    pi_posY     = (dataArr[i][j-1][1]*pi_Cmin_m - dataArr[i-1][j+1][1]*pi_Cplus_m + pi_Cplus_m*pi_Cmin_m*(dataArr[i-1][j+1][0] - dataArr[i][j-1][0]))/(pi_Cmin_m - pi_Cplus_m)
                    pi_posX     = (pi_Cplus_m*dataArr[i][j-1][0] - pi_Cmin_m*dataArr[i-1][j+1][0] + dataArr[i-1][j+1][1] - dataArr[i][j-1][1])/(pi_Cplus_m - pi_Cmin_m)
                    dataArr[i][j] = np.array([pi_posX,pi_posY,pi_theta,pi_nu,pi_mach,pi_mu,pi_Cplus_m,pi_Cmin_m,pi_Kplus,pi_Kmin])

                else: #Puntos interiores.

                    pi_Kplus    = dataArr[i][j-1][8]
                    pi_Kmin     = dataArr[i-1][j+1][9]
                    pi_theta    = (pi_Kmin + pi_Kplus)/2
                    pi_nu       = (pi_Kmin - pi_Kplus)/2
                    pi_mach     = inverse_nu(pi_nu)
                    pi_mu       = mu(pi_mach)
                    pi_Cplus_m  = math.tan(np.radians(((pi_theta + dataArr[i][j-1][2]) + (pi_mu + dataArr[i][j-1][5]))/2))
                    pi_Cmin_m   = math.tan(np.radians(((pi_theta + dataArr[i-1][j+1][2]) - (pi_mu + dataArr[i-1][j+1][5]))/2))
                    pi_posY     = (pi_Cmin_m*pi_Cplus_m*(dataArr[i][j-1][0] - dataArr[i-1][j+1][0]) + pi_Cplus_m*dataArr[i-1][j+1][1] - pi_Cmin_m*dataArr[i][j-1][1])/(pi_Cplus_m - pi_Cmin_m)
                    pi_posX     = (pi_Cplus_m*dataArr[i][j-1][0] - pi_Cmin_m*dataArr[i-1][j+1][0] - dataArr[i][j-1][1] + dataArr[i-1][j+1][1])/(pi_Cplus_m - pi_Cmin_m)
                    dataArr[i][j] = np.array([pi_posX,pi_posY,pi_theta,pi_nu,pi_mach,pi_mu,pi_Cplus_m,pi_Cmin_m,pi_Kplus,pi_Kmin])   


        if i == len(dataArr)-1: #Puntos a la salida.
            for j in range(0,len(dataArr[i])):
            
                if j == 0: #Punto en el eje.
                
                    pi_mach     = Me
                    pi_theta    = 0
                    pi_mu       = mu(pi_mach)
                    pi_nu       = nu(pi_mach)
                    pi_Kmin     = pi_theta + pi_nu
                    pi_Kplus    = pi_theta - pi_nu
                    pi_Cplus_m  = 0
                    pi_Cmin_m   = math.tan(np.radians(((pi_theta + dataArr[i-1][j+1][2]) - (pi_mu + dataArr[i-1][j+1][5]))/2))
                    pi_posY     = 0
                    pi_posX     = -dataArr[i-1][j+1][1]/pi_Cmin_m + dataArr[i-1][j+1][0]
        
                    dataArr[i][j] = np.array([pi_posX,pi_posY,pi_theta,pi_nu,pi_mach,pi_mu,pi_Cplus_m,pi_Cmin_m,pi_Kplus,pi_Kmin])
                    
                else: #Punto en la pared.
                    
                    pi_mach     = Me
                    pi_theta    = 0
                    pi_mu       = mu(pi_mach)
                    pi_nu       = nu(pi_mach)
                    pi_Kmin     = pi_theta + pi_nu
                    pi_Kplus    = pi_theta - pi_nu
                    pi_Cplus_m  = math.tan(np.radians(((pi_theta + dataArr[i][j-1][2]) + (pi_mu + dataArr[i][j-1][5]))/2))
                    pi_Cmin_m   = math.tan(np.radians((pi_theta + dataArr[i-1][-1][2])/2))
                    pi_posY     = (dataArr[i][j-1][1]*pi_Cmin_m - dataArr[i-1][-1][1]*pi_Cplus_m + pi_Cplus_m*pi_Cmin_m*(dataArr[i-1][-1][0] - dataArr[i][j-1][0]))/(pi_Cmin_m - pi_Cplus_m)
                    pi_posX     = (pi_Cplus_m*dataArr[i][j-1][0] - pi_Cmin_m*dataArr[i-1][-1][0] + dataArr[i-1][-1][1] - dataArr[i][j-1][1])/(pi_Cplus_m - pi_Cmin_m)
        
                    dataArr[i][j] = np.array([pi_posX,pi_posY,pi_theta,pi_nu,pi_mach,pi_mu,pi_Cplus_m,pi_Cmin_m,pi_Kplus,pi_Kmin])


    # Se almacenan los resultados para representar y exportar.
    results = [pd.DataFrame(dataArr[i], columns=resultCols, index=tagArr[i]) for i in range(0,n)]
    results.insert(0, p0i_results)

    xi = []
    yi = []

    for i in range(0,n):

        tempList = results[i+1]["X"].to_list()
        tempList.insert(0,results[0]["X"][i])
        xi.append(tempList)

        tempList = results[i+1]["Y"].to_list()
        tempList.insert(0,results[0]["Y"][i])
        yi.append(tempList)
    

    xCont = p0i_results['X'].to_list() + [xi[i][-1] for i in range(0,n)]
    yCont = p0i_results['Y'].to_list() + [yi[i][-1] for i in range(0,n)]
    yNegCont = [-y for y in yCont]
    

    xCont.insert(0, 0)
    yCont.insert(0, Dt/2)
    yNegCont.insert(0, -Dt/2)

    f_cont = interp1d(xCont, yCont, kind='linear')
    xList = np.linspace(xCont[0],xCont[-1],n)
    M2_guess = np.linspace(1,3,len(xList))

    Di = np.zeros(len(xList))
    ei = np.zeros(len(xList))
    M2i = np.zeros(len(xList))
    P2iMOC = np.zeros(len(xList))

    for i in range(0,len(xList)):
        Di[i] = 2*f_cont(xList[i])
        ei[i] = Di[i]/Dt
        M2i[i] = inverse_function_AR(ei[i],M2_guess[i])
        P2iMOC[i] = P1/(1+((gamma-1)/2)*M2i[i]**2)**(gamma/(gamma-1))

    pDist = interp1d(xList, P2iMOC, kind='linear')
    pDist_min, pDist_max = pDist(xList[-1]), pDist(xList[0])
    colormap = cm.get_cmap('coolwarm',1024)

    rc('text', usetex=True)
    rc('font', family='lmodern')

    fig, ax = plt.subplots()

    for xTemp in np.arange(0,xList[-1],0.0001):
        ax.plot([xTemp,xTemp],[0,f_cont(xTemp)],color=colormap((pDist(xTemp) - pDist_min) / (pDist_max - pDist_min)))
        ax.plot([xTemp,xTemp],[0,-f_cont(xTemp)],color=colormap((pDist(xTemp) - pDist_min) / (pDist_max - pDist_min)))
        plt.axis('equal')

    sm = ScalarMappable(cmap=colormap)
    sm.set_array(pDist(xList))
    cbar = plt.colorbar(sm)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2)) 
    cbar.formatter = formatter
    cbar.update_ticks()

    for i in range(0,n):
        ax.plot(xi[i], yi[i],'black',linestyle = '--',linewidth = 0.2)
        ax.plot(xi[i], [-y for y in yi[i]],'black',linestyle = '--',linewidth = 0.2)
        plt.axis('equal')

    ax.plot(xCont, yCont,'black',linewidth = 3)
    ax.plot(xCont, yNegCont,'black',linewidth = 3)

    ax.set_xlabel(r'x(m)', fontsize=16)
    ax.set_ylabel(r'y(m)', fontsize=16)
    ax.set_title(r'Tobera de Minima Longitud (MOC)', fontsize=18)
    cbar.set_label(r'Presion de Expansion (Pa)', fontsize=12, labelpad=10)


    plt.axis('equal')
    ax.xaxis.grid(linestyle='dashed', linewidth=0.5)
    ax.yaxis.grid(linestyle='dashed', linewidth=0.5)
    fig.tight_layout()

    plt.savefig('Graficos\MOC5_dibujo.png', dpi=300)
    plt.show()

    return pd.DataFrame(list(zip(xCont,yCont)), columns=["X","Y"]).to_csv('ToberasCSV\MOCaprox.csv',index=False),xCont,yCont

iter = MOC_BELL(M2,100,Dt,gamma,0.1)

# Se corrige el problema cuasi-unidemnsional con la nueva relacion de areas.
Dt_2d = iter[-1][0]*2
D2_2d = iter[-1][-1]*2
e_2d = D2_2d/Dt_2d
D2r = MOC_BELL(M2,100,Dt,gamma,0.1)[2][-1]*2
A2r = np.pi*(D2r/2)**2
er = D2r**2/Dt**2

M2r = inverse_function_AR(er,M2)
P2r = P1Result/(1+((gamma-1)/2)*M2r**2)**(gamma/(gamma-1))
v2r = np.sqrt(((2*gamma)/(gamma-1))*R*T1*(1-(P2r/P1)**((gamma-1)/gamma)))
Tr  = (G*v2r + (P2r - P3)*(np.pi*(D2r/2)**2))/9.81
T2r = T1/(1+((gamma-1)/2)*M2r**2)

Cfr  =   (Tr*9.81)/(At*P1)
Ispr = cStar*Cfr/9.81


MOC_resultIndex = [

    "Dt (1D) (m)",
    "At (1D) (m^2)",
    "D2 (1D) (m)",
    "A2 (1D) (m^2)", 
    "Dt (2D) (m)",
    "At (2D) (m^2/m)",
    "D2 (2D) (m)",
    "A2 (2D) (m^2/m)",

    "AR (1D)",
    "AR (2D)",

    "T (kg)",
    "G (kg/s)",
    "M2",
    "V2 (m/s)",
    "T2 (K)",
    "P1 (Pa)",
    "P2 (Pa)",
    "NPR",

    "Cf",
    "C* (m/s)",
    "Isp (s)"

    ]

MOC_results = np.round(np.array([Dt,At,D2r,A2r,Dt_2d,Dt_2d,D2_2d,D2_2d,er,e_2d,Tr,G,M2r,v2r,T2r,P1,P2r,P1/P3,Cfr,cStar,Ispr]), decimals=8)
MOC_results = MOC_results.tolist()

MOC_df = pd.DataFrame(MOC_results,columns=["MOC"])
