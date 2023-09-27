#TOBERA DE LAVAL AXISIMETRICA POR EL METODO DE RAO


# Problema a resolver ccon las ecuaciones de un flujo cuasi-unidemsional.
P1 = P1Result #Decision del punto de operacion.
P2  =   P3
v2  =   np.sqrt(((2*gamma)/(gamma-1))*R*T1*(1-(P2/P1)**((gamma-1)/gamma)))
G   =   regressionRate(P1,coefs[0],coefs[1])*Aref*rho
corr_factor = 0.5*(1+math.cos(np.radians(6)))
T   =   (G*v2/9.81)*corr_factor
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



# Definicion del metodo de Rao como una funcion de dos parametros
def BELL_APROX(Dt,D2):

    R2 = D2/2
    Rt = Dt/2 
    aRatio = R2**2/Rt**2
    Rchamber = 1.25*R2

    # Los angulos se eligen en base a estandares de disenno.
    theta_t = 30
    theta_e = 6
    percent_CLN = 0.8


    theta_0 = 3*np.pi/2
    theta_t = np.radians(theta_t)
    theta_e = np.radians(theta_e)

    mt = math.tan(theta_t)
    me = math.tan(theta_e)

    # El angulo de 15 grados tambien es un estandar.
    lCone = (R2-Rt)/math.tan(np.radians(15))        
    lBell = percent_CLN*lCone

    leftRt = 1.5*Rt
    rightRt = 0.4*Rt

    cLCircle = [0,2.5*Rt]
    cRCircle = [0,1.4*Rt]


    def leftCircle(theta):
        return [cLCircle[0] + leftRt*math.cos(theta), cLCircle[1] + leftRt*math.sin(theta),]

    def rightCircle(theta):
        return [cRCircle[0] + rightRt*math.cos(theta), cRCircle[1] + rightRt*math.sin(theta),]

    pE = [lBell,R2]
    pI = rightCircle(theta_t + theta_0)

    n1 = pI[1] - mt*pI[0]
    n2 = pE[1] - me*pE[0]

    pQ = [(n2-n1)/(mt-me), mt*((n2-n1)/(mt-me))+n1]

    def bellFunc(t):
        return [((1-t)**2)*pI[0] + 2*(1-t)*t*pQ[0] + (t**2)*pE[0], ((1-t)**2)*pI[1] + 2*(1-t)*t*pQ[1] + (t**2)*pE[1]]


    leftCirclePoints = np.array([leftCircle(i) for i in np.arange(np.pi,theta_0,0.01)])
    rightCirclePoints = np.array([rightCircle(i) for i in np.arange(theta_0,theta_0+theta_t,0.001)])
    bellPoints = np.array([bellFunc(i) for i in np.arange(0,1,0.001)])

    return pd.DataFrame(np.concatenate((rightCirclePoints, bellPoints), axis=0),columns=["X","Y"]).to_csv('ToberasCSV\\bellAprox.csv',index=False),rightCirclePoints,bellPoints

# Se llama a la funcion para exportar los puntos en un csv.
BELL_APROX(Dt,D2)

# Se llama a la funcion para guardar los puntos calculados.
rcp = BELL_APROX(Dt,D2)[1]
bp  = BELL_APROX(Dt,D2)[2]

# Se separan los vectores bidimensionales en formato unidemensional.
rcp_x = rcp[:,0]
rcp_y = rcp[:,1]
bp_x  = bp[:,0]
bp_y  = bp[:,1]

# Se juntan las dos secciones de la tobera.
nx = np.concatenate([rcp_x,bp_x])
ny = np.concatenate([rcp_y,bp_y])

# A partir de aqui se define todo lo necesario para crear y exportar la imagen.
Ai = np.zeros(len(nx))
ei = np.zeros(len(nx))
M2i = np.zeros(len(nx))
P2iBell = np.zeros(len(nx))

M2_guess = np.linspace(1,3,len(ny))


for i in range(0,len(ny)):
    Ai[i] = np.pi*ny[i]**2
    ei[i] = Ai[i]/At
    M2i[i] = inverse_function_AR(ei[i],M2_guess[i])
    P2iBell[i] = P1/(1+((gamma-1)/2)*M2i[i]**2)**(gamma/(gamma-1))
    

rc('text', usetex=True)
rc('font', family='lmodern')

fig, ax = plt.subplots()

pDist = P2iBell
pDist_min, pDist_max = np.min(pDist), np.max(pDist)
colormap = cm.get_cmap('coolwarm')

for i in np.arange(0,len(ny),10):
    ax.plot([nx[i],nx[i]],[0,ny[i]],color=colormap((pDist[i] - pDist_min) / (pDist_max - pDist_min)))
    ax.plot([nx[i],nx[i]],[0,-ny[i]],color=colormap((pDist[i] - pDist_min) / (pDist_max - pDist_min)))
    plt.axis('equal')

sm = ScalarMappable(cmap=colormap)
sm.set_array(pDist)
cbar = plt.colorbar(sm)

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2)) 
cbar.formatter = formatter
cbar.update_ticks()

ax.plot(rcp[:,0],rcp[:,1],'black',linewidth=3)
ax.plot(bp[:,0],bp[:,1],'black',linewidth=3)

ax.plot(rcp[:,0],-1*rcp[:,1],'black',linewidth=3)
ax.plot(bp[:,0],-1*bp[:,1],'black',linewidth=3)

ax.set_xlabel(r'x(m)', fontsize=16)
ax.set_ylabel(r'y(m)', fontsize=16)

ax.set_title(r'Tobera de Aproximacion Parabolica (RAO)', fontsize=18)
cbar.set_label(r'Presion de Expansion (Pa)', fontsize=12, labelpad=10)

plt.axis('equal')
ax.xaxis.grid(linestyle='dashed', linewidth=0.5)
ax.yaxis.grid(linestyle='dashed', linewidth=0.5)
fig.tight_layout()

plt.savefig('Graficos\BN5_dibujo.png', dpi=300)
plt.show()


# Aqui se recogen los resultados numericos de resolver el problema cuasi-unidimensional.
BN_resultIndex = [

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

BN_results = np.array([
    round(Dt,8),
    round(At,8),
    round(D2,8),
    round(A2,8),
    "-","-","-","-",
    round(e,8),"-",
    round(T,8),
    round(G,8),
    round(M2,8),
    round(v2,8),
    round(T2,8),
    round(P1,8),
    round(P2,8),
    round(P1/P2,8),
    round(Cf,8),
    round(cStar,8),
    round(Isp,8)])

BN_results = BN_results.tolist()

BN_df = pd.DataFrame(BN_results,columns=["BN"])
