# TOBERA AEROSPIKE POR EL METODO DE ANGELINO

# Se definen variables de partida
T1 = 1763
P3 = 91000
gamma = 1.2529712919314542 
R = 349.52459961214936 
R1 = 0.0075
h = 0.08
Aref = 2*np.pi*R1*h
rho = 1667

# En este caso ya es necesario introducir los datos del propelente a usar en el experimento.
regressionData = np.array([
    [68947.6,0.004],
    [344738,0.006],
    [689476,0.0075],
    [1.37895*10**6,0.0095],
    [2.41317*10**6,0.01125],
    [3.10264*10**6,0.013],
    [3.79212*10**6,0.012],
    [4.48159*10**6,0.013],
    [4.82633*10**6,0.0135],
    [5.17107*10**6,0.014],
    [5.86055*10**6,0.014],
    [5.86055*10**6,0.0155],
    [7.58424*10**6,0.015],
    [8.61845*10**6,0.0165],
    [9.30793*10**6,0.0175],
    [1.06869*10**7,0.017]
])

# Funcion para calcular el ritmo de quemado.
def regressionRate(P,a,n):
    return a*P**n
coefs, red = sc.optimize.curve_fit(regressionRate,regressionData[:,0],regressionData[:,1])

# Funcion para calcular la relacion de areas en funcion del mach.
def fe(M2):
    return (1/M2)*((2/(gamma+1))*(1+((gamma-1)/2)*M2**2))**((gamma+1)/(2*(gamma-1)))

# Funcion para calcular el mach en funcion de la relacion de areas.
def inverse_function_AR(y,Mg):
    equation = lambda M2: fe(M2) - y
    M2_initial_guess = Mg 
    M2_approximation, = fsolve(equation, M2_initial_guess)

    return M2_approximation

# Funcion que aplica el metodo de Angelino
def MOC_AEROSPIKE(P1,gamma,n,eta_b):

    # Se definen funciones auxiliares para calcular todo lo necesario durante el proceso.
    resultCols = ["X","Y","Theta","Nu","M","Mu","C+(dy/dx)","C-(dy/dx)","K+","K-"]
    prandtlMayerCols = ["Mach","Nu(M)"]

    def M2(P1):
        return np.sqrt((2/(gamma-1))*((P1/P3)**((gamma-1)/gamma)-1))

    def AR(_M_):
        return (1/_M_)*((2/(gamma+1))*(1+((gamma-1)/2)*_M_**2))**((gamma+1)/(2*(gamma-1)))

    def nu(_M_):
        return (np.sqrt(gK)*math.atan(np.sqrt((1/gK)*(_M_**2-1))) - math.atan(np.sqrt(_M_**2-1)))*(180/np.pi)

    Me = M2(P1)
    gK = (gamma+1)/(gamma-1)
    Mmax = 10

    Tt = 2*T1/(gamma+1)
    Pt = P1/(((gamma+1)/2)**(gamma/(gamma-1)))
    vt = np.sqrt(gamma*R*Tt)

    Gr = regressionRate(P1,coefs[0],coefs[1])*Aref*rho
    At = ((R*Tt*Gr)/(Pt*vt))/math.sin(np.radians(nu(Me)))
    At_axial = ((R*Tt*Gr)/(Pt*vt))
    e  = AR(M2(P1))
    Ae = At*e
    re = np.sqrt(Ae/np.pi)

    prandtlMayerCols = ["Mach","Nu(M)"]
    prandtlMayerArr = []

    for M in np.arange(1,Mmax,0.0001):
        prandtlMayerArr.append([M,nu(M)])
    prandtlMayerArr.append([Mmax,nu(Mmax)])

    prandtlMayerData = pd.DataFrame(prandtlMayerArr,columns=prandtlMayerCols)

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

    def mu(_M_):
        return math.asin(1/_M_)*(180/np.pi)

    def char_ad_lenght(eta_b,_M_,_alpha_):
        return (1 - (1 - (AR(_M_)*(1 - eta_b**2)*_M_*(math.sin(np.radians(_alpha_))/AR(Me))))**(1/2))/(math.sin(np.radians(_alpha_)))


    #Iniciacion del calculo.

    muList = np.zeros(n)
    lenghtList = np.zeros(n)
    alphaList = np.zeros(n)
    parametricAlpha = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    machList = np.linspace(1,Me,n)
    P2i = np.zeros(n)
    theta2i = np.zeros(n)

    for i in range(0,n):

        discr = 1 - (AR(machList[i])/AR(Me))*(1 - eta_b**2)*machList[i]*math.sin(np.radians(mu(machList[i]) + nu(Me) - nu(machList[i])))

        if discr > 0:
            y[i] = re*np.sqrt(discr)
        else:
            y[i] = 0

        x[i] = (re - y[i])/math.tan(np.radians(mu(machList[i]) + nu(Me) - nu(machList[i])))
        P2i[i] = P1/(1+((gamma-1)/2)*machList[i]**2)**(gamma/(gamma-1))


    Lt = np.sqrt(x[0]**2 + (re - y[0])**2)
    Atg = np.pi*(re**2-y[0]**2)/math.sin(np.radians(nu(Me)))
    Aeg = np.pi*(re**2-y[-1]**2)

    theta1 = np.degrees(math.atan(x[0]/(re - y[0])))
    Gt = (Pt*vt*Atg)/(R*Tt)
    F1 = (Gr*vt + (Pt - P3)*Atg)*math.cos(np.radians(theta1))/9.81


    for i in range(0,n-1):
        theta2i[i] = np.degrees(math.atan((x[i+1]-x[i])/(y[i]-y[i+1])))
    theta2i[-1] = 90

    f2 = [(((P2i[i]-P3)*np.pi*np.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)*(y[i+1]+y[i]))*math.cos(math.atan((y[i+1]-y[i])/(x[i+1]-x[i])))) for i in range(0,n-1)]
    F2 = sum(f2)/9.81

    T = F1 + F2

    Cf  =   (T*9.81)/(At_axial*P1)
    cStar = P1*At_axial/Gr
    Isp = cStar*Cf/9.81

    results = pd.DataFrame(list(zip(x,y,machList,P2i,theta2i)), columns=["X","Y","Mach","P2i",'theta2i'])

    return T,F1,F2,theta1,pd.DataFrame(list(zip(x,y)), columns=["X","Y"]),Lt,Ae/At,Me,Gr,Gt,At,Ae,re,Atg,Aeg,f2,Cf,cStar,Isp,P2i,At_axial,vt,Tt,Pt

#Representacion de Tobera Aerospike y exportacion de resultados.

# Magnitudes Axiales
At = MOC_AEROSPIKE(P1Result,gamma,100,0)[20]
D2  = MOC_AEROSPIKE(P1Result,gamma,100,0)[12]*2
Rt = np.sqrt((D2/2)**2 - At/np.pi)
A2 = MOC_AEROSPIKE(P1Result,gamma,100,0)[11]
Lt = (D2/2) - Rt
e = A2/At


# Magnitudes Reales
Lt_2d = MOC_AEROSPIKE(P1Result,gamma,100,0)[5]
At_2d = MOC_AEROSPIKE(P1Result,gamma,100,0)[10]
D2_2d = MOC_AEROSPIKE(P1Result,gamma,100,0)[12]*2
A2_2d = MOC_AEROSPIKE(P1Result,gamma,100,0)[11]
e_2d = A2_2d/At_2d

T = MOC_AEROSPIKE(P1Result,gamma,100,0)[0]
G = MOC_AEROSPIKE(P1Result,gamma,100,0)[8]
M2 = 1
V2 = MOC_AEROSPIKE(P1Result,gamma,100,0)[21]
T2 = MOC_AEROSPIKE(P1Result,gamma,100,0)[22]
P1 = P1Result
P2 = MOC_AEROSPIKE(P1Result,gamma,100,0)[23]
Cf = MOC_AEROSPIKE(P1Result,gamma,100,0)[16]
cStar = MOC_AEROSPIKE(P1Result,gamma,100,0)[17]
Isp =  MOC_AEROSPIKE(P1Result,gamma,100,0)[18]


AS_resultIndex = [

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

AS_results = np.round(np.array([Lt,At,D2,A2,Lt_2d,At_2d,D2_2d,A2_2d,e,e_2d,T,G,M2,V2,T2,P1,P2,P1/P2,Cf,cStar,Isp]), decimals=8)
AS_results = AS_results.tolist()

AS_df = pd.DataFrame(AS_results,columns=["AS"])

x   = MOC_AEROSPIKE(P1Result,gamma,100,0)[4]["X"]
y   = MOC_AEROSPIKE(P1Result,gamma,100,0)[4]["Y"]
De  = MOC_AEROSPIKE(P1Result,gamma,100,0)[12]*2

rc('text', usetex=True)
rc('font', family='lmodern')

fig, ax = plt.subplots()

pDist = MOC_AEROSPIKE(P1Result,gamma,100,0)[19]
pDist_min, pDist_max = np.min(pDist), np.max(pDist)

colormap = cm.get_cmap('coolwarm')

for i in range(0,100):
    ax.plot([0,x[i]],[De/2,y[i]],color=colormap((pDist[i] - pDist_min) / (pDist_max - pDist_min)))
    ax.plot([0,x[i]],[-De/2,-y[i]],color=colormap((pDist[i] - pDist_min) / (pDist_max - pDist_min)))
    plt.axis('equal')
    
sm = ScalarMappable(cmap=colormap)
sm.set_array(pDist)
cbar = plt.colorbar(sm)

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2)) 
cbar.formatter = formatter
cbar.update_ticks()

    
ax.plot(x,y,'black',linewidth=3)
ax.plot(x,[-i for i in y],'black',linewidth=3)
ax.plot([x[99],x[99]],[y[99],-y[99]],'black',linewidth=3)


ax.set_xlabel(r'x(m)', fontsize=16)
ax.set_ylabel(r'y(m)', fontsize=16)

ax.set_title(r'Tobera Aerospike (Angelino)', fontsize=18)
cbar.set_label(r'Presion de Expansion (Pa)', fontsize=12, labelpad=10)

plt.axis('equal')
ax.xaxis.grid(linestyle='dashed', linewidth=0.5)
ax.yaxis.grid(linestyle='dashed', linewidth=0.5)
fig.tight_layout()

plt.savefig('Graficos\AS5_dibujo.png', dpi=300)
plt.show()

pd.DataFrame(np.array(list(zip(x,y))),columns=["X","Y"]).to_csv('ToberasCSV\spikeAprox.csv',index=False)


