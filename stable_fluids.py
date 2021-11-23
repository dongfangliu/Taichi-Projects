import taichi as ti
from taichi.lang.ops import exp, floor, random, sqrt
import numpy as np
# define flow field size
x_resol = 512
y_resol = 512
dx = 0.1
density=1000.0
dt = 0.03

N = (x_resol-2)*(y_resol-2)

time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)

force_strength= 1000.0
force_radius = x_resol/2.
width = x_resol*dx
height = y_resol*dx


ti.init()

u = ti.field(dtype = ti.f32,shape = (x_resol+1,y_resol))
v = ti.field(dtype = ti.f32,shape = (x_resol,y_resol+1))
new_u = ti.field(dtype = ti.f32,shape = (x_resol+1,y_resol))
new_v = ti.field(dtype = ti.f32,shape = (x_resol,y_resol+1))
pressure = ti.field(dtype = ti.f32,shape = (x_resol,y_resol))
vel = ti.Vector.field(n=2,dtype = ti.f32,shape = (x_resol,y_resol))
vel_norm = ti.field(dtype = ti.f32,shape = (x_resol,y_resol))
_dye_buffer = ti.Vector.field(3, float, shape=(x_resol, y_resol))
_new_dye_buffer = ti.Vector.field(3, float, shape=(x_resol, y_resol))


rhs = ti.field(dtype = ti.f32,shape = N)

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

u_pair =TexPair(u,new_u)
v_pair = TexPair(v,new_v)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)

@ti.func
def grid_center_velocity(u:ti.template(),v:ti.template(),i:ti.int8,j:ti.int8):
    return ti.Vector([0.5*(u[i,j]+u[i-1,j]),0.5*(v[i,j]+v[i,j-1])])

@ti.func
def interpolate_velocity(u:ti.template(),v:ti.template(),p):
    p[0] = max(0.0,min(width,p[0]))
    p[1] = max(0.0,min(height,p[1]))
    i = int( floor(p[0]/dx))
    j = int(floor(p[1]/dx))
    tx = p[0]/dx-i
    ty = p[1]/dx-j
    u0 = grid_center_velocity(u,v,i,j)
    u1 = grid_center_velocity(u,v,i+1,j)
    u2 = grid_center_velocity(u,v,i,j+1)
    u3 = grid_center_velocity(u,v,i+1,j+1)
    u01= (1-tx)*u0+tx*u1
    u23= (1-tx)*u2+tx*u3
    return (1-ty)*u01+ty*u23

@ti.func
def interpolate_quantity(phi:ti.template(),p):
    p[0] = max(0.0,min(width,p[0]))
    p[1] = max(0.0,min(height,p[1]))
    i = int( floor(p[0]/dx))
    j = int(floor(p[1]/dx))
    tx = p[0]/dx-i
    ty = p[1]/dx-j
    u0 = phi[i,j]
    u1 = phi[i+1,j]
    u2 = phi[i,j+1]
    u3 = phi[i+1,j+1]
    u01= (1-tx)*u0+tx*u1
    u23=(1-tx)*u2+tx*u3
    return (1-ty)*u01+ty*u23

@ti.kernel 
def advect_quantity(u_in:ti.template(),v_in:ti.template(),phi_in:ti.template(),phi_out:ti.template(),):
    # advection
    for i,j in ti.ndrange((1,x_resol-1),(1,y_resol-1)):
        u1  =grid_center_velocity(u,v,i,j)
        p = ti.Vector([i,j])*dx
        p1 = p -0.5*dt*u1
        u2 = interpolate_velocity(u_in,v_in,p1)
        p2 = p1-0.75*dt*u2
        u3 = interpolate_velocity(u_in,v_in,p2)
        p-= dt * ((2 / 9) * u1 + (1 / 3) * u2 + (4 / 9) * u3)
        phi_out[i,j]=interpolate_quantity(phi_in,p)


@ti.kernel  
def apply_forces(u:ti.template(),v:ti.template(),imp_data: ti.ext_arr()):
    # external forces
    for i,j in ti.ndrange((1,x_resol-1),(1,y_resol-1)):
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])


        dx, dy = ( float(i) - 0.5 - omx), ( float(j)  - omy)
        d2 = dx * dx + dy * dy
        factor = ti.exp(-d2 / force_radius)
        momentum = (mdir * force_strength * factor ) * dt
        # scale by phsics unit
        momentum*=dx
        
       
        uu = u[i,j]
        vv = v[i,j]
        u[i,j]=uu+momentum[0]
        v[i,j]=vv+momentum[1]

        dc = dyes_pair.cur[i, j]
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (y_resol / 15)**2)) * ti.Vector(
                [imp_data[4], imp_data[5], imp_data[6]])

        dyes_pair.cur[i, j] = dc

@ti.kernel 
def fill_pressure_solve_lhs(A:ti.linalg.SparseMatrixBuilder()):
    for i,j in ti.ndrange((1,x_resol-1),(1,y_resol-1)):
        res = y_resol-2
        # scale = dt/density/dx/dx
        scale=1.0
        row = (i-1)*res+j-1
        center = 0.0
        if i-1!=0:
            A[row,row-res] +=-scale
            center+=scale
        if i+1!=x_resol-1:
            A[row,row+res] +=-scale
            center+=scale
        if j-1!=0:
            A[row, row - 1] += -scale
            center += scale
        if j+1!=y_resol-1:
            A[row, row + 1] += -scale
            center += scale
        A[row,row]+=center


@ti.kernel
def fill_pressure_solve_rhs(u:ti.template(),v:ti.template()):
    for i,j in ti.ndrange((1,x_resol-1),(1,y_resol-1)):
        # scale = 1.0/dx
        scale = dx*density/dt
        ind= (i-1)*(y_resol-2)+j-1
        rhs[ind]=-(u[i+1,j]-u[i,j]+v[i,j+1]-v[i,j])*scale
        if i-1==0:
            rhs[ind]-=scale*(u[i,j]-0)
        if i+1==x_resol-1:
            rhs[ind]-=scale*(0-u[i+1,j])
        if j-1==0:
            rhs[ind]-=scale*(v[i,j]-0)
        if j+1==y_resol-1:
            rhs[ind]-=scale*(0-v[i,j+1])

@ti.kernel 
def pressure_vec_to_mat(p_in:ti.ext_arr(),p_out:ti.template()):
    for i,j in ti.ndrange((1,x_resol-1),(1,y_resol-1)):
        p_out[i,j]=p_in[(i-1)*(y_resol-2)+j-1]



@ti.kernel
def pressure_update(u:ti.template(),v:ti.template(),pressure:ti.template(),vel:ti.template()):
    # pressure update
    for i,j in ti.ndrange((1,x_resol-1),(1,y_resol-1)):
        scale = dt/(density*dx)
        if i==1 :
            u[i,j]=0
        else:
            u[i,j]-=scale*(pressure[i,j]-pressure[i-1,j])
        if j==1 :
            v[i,j]=0
        else:
            v[i,j]-=scale*(pressure[i,j]-pressure[i,j-1])
        vel[i,j]=grid_center_velocity(u,v,i,j)
        vel_norm[i,j] = sqrt(vel[i,j][0]*vel[i,j][0]+vel[i,j][1]*vel[i,j][1])
    

K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
fill_pressure_solve_lhs(K)
A = K.build()
solver = ti.linalg.SparseSolver()
solver.analyze_pattern(A)
solver.factorize(A)

def step(mouse_data):
    # advect_velocity(u_pair.cur,v_pair.cur,u_pair.nxt,v_pair.nxt)
    advect_quantity(u_pair.cur,v_pair.cur,u_pair.cur,u_pair.nxt)
    advect_quantity(u_pair.cur,v_pair.cur,v_pair.cur,v_pair.nxt)
    advect_quantity(u_pair.cur,v_pair.cur,dyes_pair.cur,dyes_pair.nxt)
    u_pair.swap()
    v_pair.swap()
    dyes_pair.swap()
    apply_forces(u_pair.cur,v_pair.cur,mouse_data)
    fill_pressure_solve_rhs(u_pair.cur,v_pair.cur)
    p_cur = solver.solve(rhs)
    pressure_vec_to_mat(p_cur,pressure)
    pressure_update(u_pair.cur,v_pair.cur,pressure,vel)

   

    

class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array([gui.get_cursor_pos()[0]*x_resol,gui.get_cursor_pos()[1]*y_resol], dtype=np.float32) 
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data

def reset():
    u_pair.cur.fill(0)
    v_pair.cur.fill(0)
    pressure.fill(0)

gui = ti.GUI('Stable Fluid', (x_resol, y_resol))
md_gen = MouseDataGen()
paused = False
visualize_d = False  #visualize dye (default)
visualize_v = True  #visualize velocity=
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        if e.key == ti.GUI.ESCAPE:
            break
        elif e.key == 'r':
            paused = False
            reset()
        elif e.key == 'p':
            paused = not paused
        elif e.key == 'v':
            visualize_v = True
            visualize_d = False
        elif e.key == 'd':
            visualize_d = True
            visualize_v = False

    if not paused:
        mouse_data = md_gen(gui)
        step(mouse_data)
    if visualize_v:# gui.set_image(vel.to_numpy() * 0.01 + 0.5)
        gui.set_image(vel_norm.to_numpy() * 0.05)
    elif visualize_d:
        gui.set_image(dyes_pair.cur)
    gui.show()
