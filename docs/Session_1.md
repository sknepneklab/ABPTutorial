# Session 1
## Implementing a 2D simulation of Active Brownian Particles (ABP) in Python.

### Overview of the problem

#### Description of the model

Let us consider a two-dimensional system consisting of $N$ identical disks of radius $a$. The instantaneous position of disk $i$ is given by the radius vector $\mathbf{r}_i = x_i\mathbf{e}_x+y_i\mathbf{e}_y$. In addition, each disk is polar and its polarity is described by a vector $\mathbf{n}_i = \cos(\vartheta_i)\mathbf{e}_x + \sin(\vartheta_i)\mathbf{e}_y$, where $\vartheta_i$ is angle between $\mathbf{n}_i$ and the $x-$axis of the laboratory reference frame. It is immediately clear that $\left|\mathbf{n}_i\right| = 1$.

Disks are assumed to be soft, i.e., they repel each other if they overlap. The interaction force is therefore short-range and, for simplicity, we assume it to be harmonic, i.e., the force between disks $i$ and $j$ is
\begin{equation}
\mathbf{F}_{ij} = 
\begin{cases} 
    k\left(2a - r_{ij}\right)\hat{\mathbf{r}}_{ij} & \text{if } r_{ij} \le 2a \\ 
    0 & \text{otherwise}
\end{cases},\label{eq:force}
\end{equation}
where $k$ is the spring stiffness constant, $\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$, $r_{ij} = \left|\mathbf{r}_{ij}\right|$ is the distance between particles $i$ and $j$ and $\hat{\mathbf{r}}_{ij} = \frac{\mathbf{r}_{ij}}{r_{ij}}$. 

Furthermore, overlapping disk are assumed to experience torque that acts to align their polarities. Torque on disk $i$ due to disk $j$ is 
\begin{equation}
\boldsymbol{\tau}_{ij} = 
\begin{cases} 
    J\mathbf{n}_i\times\mathbf{n}_j & \text{if } r_{ij} \le 2a \\ 
    0 & \text{otherwise}
\end{cases}\label{eq:torque},
\end{equation}
where $J$ is the alignment strength. Note that since we are working in two dimensions, $\boldsymbol{\tau}_{ij} = \tau_{ij}\mathbf{e}_z$. It is easy to show that $\tau_{ij} = -J\sin\left(\vartheta_i-\vartheta_j\right)$. 

Finally, each disk is self-propelled along the direction of its vector with a force 
\begin{equation}
\mathbf{F}_i = \alpha \mathbf{n}_i,\label{eq:spforce}
\end{equation}
where $\alpha$ is the magnitude of the self-propulsion force.

Key ingredients of the model are show in the figure below:

<div align="center">
<img src="./system.png" style="width: 800px;"/>
</div>

#### Equations of motion
    
In the overdamped limit, all inertial effects are neglected and the equations of motion become simply force (and torque) balance equations. For the model defined above we have,
\begin{eqnarray}
    \dot{\mathbf{r}}_i & = & v_0 \mathbf{n}_i + \frac{1}{\gamma_t}\sum_j \mathbf{F}_{ij} + \boldsymbol{\xi}^{t}_{i}\\
    \dot{\vartheta}_i & = & -\frac{1}{\tau_r}\sum_j\sin\left(\vartheta_i-\vartheta_j\right) + \xi_i^r.\label{eq:motion_theta}
\end{eqnarray}
In previous equations we introduced the translational and rotational friction coefficients, $\gamma_t$ and $\gamma_r$, respectively and defined $v_0 = \frac{\alpha}{\gamma_t}$ and $\tau_r = \frac{\gamma_r}{J}$. Note that $v_0$ has units of velocity (hence we interpret it as the self-propulsion speed) while $\tau_r$ has units of time (hence we interpret it as the polarity alignment time scale). $\boldsymbol{\xi}_i^t$ is the white noise, which we here assume to be thermal noise (in general, this assumption is not required for a system out of equilibrium), i.e., $\langle\xi_{i,\alpha}\rangle=0$ and $\langle \xi_{i,\alpha}^t(t)\xi_{j,\beta}^t(t^\prime)\rangle = 2\frac{k_BT}{\gamma_t}\delta_{ij}\delta_{\alpha\beta}\delta(t-t^\prime)$, with $\alpha, \beta \in \{x, y\}$ and $T$ being temperature. Similarly, $\xi_i^r$ is the rotational noise, with $\langle\xi_i^r\rangle = 0$ and $\langle \xi_i^r(t)\xi_j^r(t^\prime)\rangle = 2D_r\delta_{ij}\delta(t-t^\prime)$, where we have introduced the rotational diffusion coefficient $D_r$.

<div align="center">
<font size="5" color="#990000"><b>The purpose of this tutorial is to develop a computer code that solves these equations of motion.</b></font>
</div>

<div class="alert alert-block alert-warning">
    <b>Note:</b> In this session we will sacrifice performance for clarity. Codes presented here are not optimized and quite often the implementation is very inefficient. This has been done deliberately in order not to obfuscate the key concepts. 

Our design philosophy, however, is to split the problem into a set of loosely coupled modules. This makes testing and maintenance simple.
</div>

### Overview of a particle-based simulation

A typical particle-based simulation workflow consists of three steps:

1. Creating the initial configuration; 
2. Executing the simulation; 
3. Analyzing the results.

The standard workflow is: step 1 feeds into step 2, which, in turn feeds into step 3. 

<div class="alert alert-block alert-info">
Depending on the approach and the problem at hand, sometimes these three steps are done by the same code. In line with our key design philosophy of keeping things as detached from each other as possible, in this tutorial we treat these three steps as independent. The communication between different steps will be based on a shared file format, i.e., a code written to perform step 1 will produce output files in a format that the code in step 2 can directly read, etc.
</div>

#### Creating the initial configuration

In the first step, we need to generate the system we would like to study, i.e. we need to create the initial configuration for our simulation. Sometimes, as it is the case here, this is a fairly simple task. However, creating a proper initial configuration can also be a challenging task in its own right that requires a set of sophisticated tools to do. A good example would be setting up a simulation of several complex biomolecules.

Regardless of the complexity, typically the result of this step is one (or several) text (sometimes binary) files that contain information such as the size of the simulation box, initial positions and velocities of all particles, particle connectivities, etc.

In this tutorial, we will use a set of simple Python scripts (located in *Python/pymd/builder* directory) to create the initial configuration that will be saved as a single JSON file.

For example, let us build the initial configuration with a square simulation box of size $L=50$ with the particle number density $\phi=\frac{N}{L^2}=0.4$. We also make sure that centers of no two particles less than $a=1$ apart. Note that below we will assume that particles have unit radius. This means that our initial configuration builder allowed for significant overlaps between particles. In practice this is not a problem since we will be using soft-core potential and even large overlaps do not lead to excessive force. If one is to consider a system where particles interact via a potential with a strong repulsive component (e.g., the Lennard-Jones potential) the initial configuration would have to be constructed more carefully. 
<div class="alert alert-block alert-warning">
Neither $L$ not $a$ have units here. We'll get back to the issue of simulation units below.
</div>


```python
from pymd.builder import *
phi = 0.4
L = 50
a = 1.0
random_init(phi, L, rcut=a, outfile='init.json')  
```

A simple inspection of the file *init.json* confirms that $N=1000$ particles have been created in the simulation box of size $50\times50$.

<div class="alert alert-block alert-info">
    <b>Note:</b> In this tutorial we will aways assume the simulation box is orthogonal and centered at $(0,0)$. In other words, for simulation box of size $L_x=50$, $L_y=40$, all particles are located in the rectangle with corners at $(-25,-20)$, $(25,-20)$, $(25,20)$, $(-25,20)$.
</div>

#### Executing the simulation 

This step (i.e., step 2) is the focus of this tutorial. Conceptually, this step is straightforward - we need to write a program that solves the set equations for motion for a collection of $N$ particles placed in the simulation box. In practice, this is technically the most challenging step and some of the most advanced particle based codes can contain hundreds of thousands of lines of code. 

The problem is that equations of motions are coupled, i.e., we are tasked with solving the N-body problem. Therefore, any naive implementation would be so slow that it would not be possible to simulate even the simplest systems within any reasonable time. For this reason, even the most basic particle based codes have to implement several algorithms that make the problem tractable. Furthermore, any slightly more complex problem would require parallelization, which is the topic of the last session in this tutorial. 

#### Analyzing the results

Once the simulation in the step 2 has been completed, we need to perform a series of measurements to extract as much physical information about our system as possible. Although sometimes some basic analysis can be done by the main simulation code (i.e., in step 2), this is typically done a posteriori with a separate set of tools. Technically speaking, the main simulation code produces a file (or a set of files), often referred to as *trajectories*, that are loaded into the analysis code for post-processing. 

One of the key steps in the post-processing stage is visualization, e.g. making a movie of the time evolution of the system. Therefore, most simulation codes are able to output the data in multiple file formats for various types of post-processing analysis.

In this tutorial we will output the data in JSON and VTP formats. VTP files can be directly visualized with the powerful [Paraview](https://www.paraview.org/) software package. 

<div class="alert alert-block alert-info">
    <b>Note:</b> There are many excellent codes that perform step 2. For example, these are: GROMACS, LAMMPS, HOOMD Blue, ESPResSo, AMBER, DLPOLY, to name a few. Our aim is not to replace any of those codes but to showcase in a simple example the core of what these codes do.
</div>


### Writing a particle based simulation in Python

Here we outline the key parts of a modern implementation of a particle-based simulation. We use Python for its simple syntax and powerful data structures.

#### Periodic boundary conditions

Before we dive deep into the inner workings of a particle-based code, let us quickly discuss the use of the periodic boundary conditions (PBC). 

Even the most advanced particle-based simulations can simulate no more than several million particles. Most simulations are far more modest in size. A typical experimental system usually contains far more particles than it would be possible to simulate. Therefore, most simulations study a small section of an actual system and extrapolate the results onto the experimentally relevant scales. It is, therefore, important to minimize the finite size effects. One such approach is to use the  periodic boundary conditions.

The idea behind PBC is that the simulation box is assumed to have the topology of a torus. That is, a particle that leaves the simulation box through the left boundary would instantaneously reappear on the right side of the simulation box. This implies that when computing the distance between two particles one has to consider all of their periodic images and pick the shortest distance. This is called the *minimum image convention*.

The idea behind the minimum image convention is shown in the image below.

<div align="center">
    
<img src="./pbc.png" style="width: 600px;"/>
</div>

#### Key components of a particle-based simulation code

The image below shows the key components of a particle-based code and a possible design layout of how to organize them.

<div align="center">
    
<img src="./layout.png" style="width: 800px;"/>
</div>
<div class="alert alert-block alert-warning">
    <b>Note:</b> This is one of many design layouts that can be used. The one chosen here reflects our design philosophy of minimizing the interdependence of different components of the code. In addition, it naturally fits with the aim to use Python as the scripting language for controlling the simulation.</div>

#### A working example

Let us start with a working example of a full simulation. 

We read the initial configuration stored in the file *init.json* with $N=1,000$ randomly placed particles in a square box of size $L=50$. We assume that all particles have the same radius $a=1$. Further, each particle is self-propelled with the active force of magnitude $\alpha=1$ and experiences translational friction with friction coefficient $\gamma_t = 1$. Rotational friction is set to $\gamma_r = 1$ and the rotational diffusion constant to $D_r = 0.1$. Particles within the distance $d=2$ of each other experience the polar alignment torque of magnitude $J=1$.

We use the time step $\delta t = 0.01$ and run the simulation for $1,000$ time steps. We record a snapshot of the simulation once every 10 time steps.


```python
from pymd.md import *               # Import the md module from the pymd package

s = System(rcut = 3.0, pad = 0.5)   # Create a system object with neighbour list cutoff rcut = 3.0 and padding distance 0.5
s.read_init('init.json')            # Read in the initial configuration

e = Evolver(s)                      # Create a system evolver object
d = Dump(s)                         # Create a dump object

hf = HarmonicForce(s, 10.0, 2.0)    # Create pairwise repulsive interactions with the spring contant k = 10 and range a = 2.0
sp = SelfPropulsion(s, 1.0)         # Create self-propulsion, self-propulsion strength alpha = 1.0
pa = PolarAlign(s, 1.0, 2.0)        # Create pairwise polar alignment with alignment strength J = 1.0 and range a = 2.0

pos_integ = BrownianIntegrator(s, T = 0.0, gamma = 1.0)       # Integrator for updating particle position, friction gamma = 1.0 and no thermal noise
rot_integ = BrownianRotIntegrator(s, T = 0.1, gamma = 1.0)    # Integrator for updating particle oriantation, friction gamma = 1.0, "rotation" T = 0.1, D_r = 0.0 

# Register all forces, torques and integrators with the evolver object
e.add_force(hf)                    
e.add_force(sp)
e.add_torque(pa)
e.add_integrator(pos_integ)
e.add_integrator(rot_integ)
```


```python
dt = 0.01    # Simulation time step
# Run simulation for 1,000 time steps (total simulation time is 10 time units)
for t in range(1000):
  print("Time step : ", t)
  e.evolve(dt)       # Evolve the system by one time step of length dt
  if t % 10 == 0:    # Produce snapshot of the simulation once every 10 time steps
    d.dump_vtp('test_{:05d}.vtp'.format(t))
```

    Time step :  0
    Time step :  1
    Time step :  2
    Time step :  3
    Time step :  4
    Time step :  5
    Time step :  6
    Time step :  7
    Time step :  8
    Time step :  9
    Time step :  10
    Time step :  11
    Time step :  12
    Time step :  13
    Time step :  14
    Time step :  15
    Time step :  16
    Time step :  17
    Time step :  18
    Time step :  19
    Time step :  20
    Time step :  21
    Time step :  22
    Time step :  23
    Time step :  24
    Time step :  25
    Time step :  26
    Time step :  27
    Time step :  28
    Time step :  29
    Time step :  30
    Time step :  31
    Time step :  32
    Time step :  33
    Time step :  34
    Time step :  35
    Time step :  36
    Time step :  37
    Time step :  38
    Time step :  39
    Time step :  40
    Time step :  41
    Time step :  42
    Time step :  43
    Time step :  44
    Time step :  45
    Time step :  46
    Time step :  47
    Time step :  48
    Time step :  49
    Time step :  50
    Time step :  51
    Time step :  52
    Time step :  53
    Time step :  54
    Time step :  55
    Time step :  56
    Time step :  57
    Time step :  58
    Time step :  59
    Time step :  60
    Time step :  61
    Time step :  62
    Time step :  63
    Time step :  64
    Time step :  65
    Time step :  66
    Time step :  67
    Time step :  68
    Time step :  69
    Time step :  70
    Time step :  71
    Time step :  72
    Time step :  73
    Time step :  74
    Time step :  75
    Time step :  76
    Time step :  77
    Time step :  78
    Time step :  79
    Time step :  80
    Time step :  81
    Time step :  82
    Time step :  83
    Time step :  84
    Time step :  85
    Time step :  86
    Time step :  87
    Time step :  88
    Time step :  89
    Time step :  90
    Time step :  91
    Time step :  92
    Time step :  93
    Time step :  94
    Time step :  95
    Time step :  96
    Time step :  97
    Time step :  98
    Time step :  99
    Time step :  100
    Time step :  101
    Time step :  102
    Time step :  103
    Time step :  104
    Time step :  105
    Time step :  106
    Time step :  107
    Time step :  108
    Time step :  109
    Time step :  110
    Time step :  111
    Time step :  112
    Time step :  113
    Time step :  114
    Time step :  115
    Time step :  116
    Time step :  117
    Time step :  118
    Time step :  119
    Time step :  120
    Time step :  121
    Time step :  122
    Time step :  123
    Time step :  124
    Time step :  125
    Time step :  126
    Time step :  127
    Time step :  128
    Time step :  129
    Time step :  130
    Time step :  131
    Time step :  132
    Time step :  133
    Time step :  134
    Time step :  135
    Time step :  136
    Time step :  137
    Time step :  138
    Time step :  139
    Time step :  140
    Time step :  141
    Time step :  142
    Time step :  143
    Time step :  144
    Time step :  145
    Time step :  146
    Time step :  147
    Time step :  148
    Time step :  149
    Time step :  150
    Time step :  151
    Time step :  152
    Time step :  153
    Time step :  154
    Time step :  155
    Time step :  156
    Time step :  157
    Time step :  158
    Time step :  159
    Time step :  160
    Time step :  161
    Time step :  162
    Time step :  163
    Time step :  164
    Time step :  165
    Time step :  166
    Time step :  167
    Time step :  168
    Time step :  169
    Time step :  170
    Time step :  171
    Time step :  172
    Time step :  173
    Time step :  174
    Time step :  175
    Time step :  176
    Time step :  177
    Time step :  178
    Time step :  179
    Time step :  180
    Time step :  181
    Time step :  182
    Time step :  183
    Time step :  184
    Time step :  185
    Time step :  186
    Time step :  187
    Time step :  188
    Time step :  189
    Time step :  190
    Time step :  191
    Time step :  192
    Time step :  193
    Time step :  194
    Time step :  195
    Time step :  196
    Time step :  197
    Time step :  198
    Time step :  199
    Time step :  200
    Time step :  201
    Time step :  202
    Time step :  203
    Time step :  204
    Time step :  205
    Time step :  206
    Time step :  207
    Time step :  208
    Time step :  209
    Time step :  210
    Time step :  211
    Time step :  212
    Time step :  213
    Time step :  214
    Time step :  215
    Time step :  216
    Time step :  217
    Time step :  218
    Time step :  219
    Time step :  220
    Time step :  221
    Time step :  222
    Time step :  223
    Time step :  224
    Time step :  225
    Time step :  226
    Time step :  227
    Time step :  228
    Time step :  229
    Time step :  230
    Time step :  231
    Time step :  232
    Time step :  233
    Time step :  234
    Time step :  235
    Time step :  236
    Time step :  237
    Time step :  238
    Time step :  239
    Time step :  240
    Time step :  241
    Time step :  242
    Time step :  243
    Time step :  244
    Time step :  245
    Time step :  246
    Time step :  247
    Time step :  248
    Time step :  249
    Time step :  250
    Time step :  251
    Time step :  252
    Time step :  253
    Time step :  254
    Time step :  255
    Time step :  256
    Time step :  257
    Time step :  258
    Time step :  259
    Time step :  260
    Time step :  261
    Time step :  262
    Time step :  263
    Time step :  264
    Time step :  265
    Time step :  266
    Time step :  267
    Time step :  268
    Time step :  269
    Time step :  270
    Time step :  271
    Time step :  272
    Time step :  273
    Time step :  274
    Time step :  275
    Time step :  276
    Time step :  277
    Time step :  278
    Time step :  279
    Time step :  280
    Time step :  281
    Time step :  282
    Time step :  283
    Time step :  284
    Time step :  285
    Time step :  286
    Time step :  287
    Time step :  288
    Time step :  289
    Time step :  290
    Time step :  291
    Time step :  292
    Time step :  293
    Time step :  294
    Time step :  295
    Time step :  296
    Time step :  297
    Time step :  298
    Time step :  299
    Time step :  300
    Time step :  301
    Time step :  302
    Time step :  303
    Time step :  304
    Time step :  305
    Time step :  306
    Time step :  307
    Time step :  308
    Time step :  309
    Time step :  310
    Time step :  311
    Time step :  312
    Time step :  313
    Time step :  314
    Time step :  315
    Time step :  316
    Time step :  317
    Time step :  318
    Time step :  319
    Time step :  320
    Time step :  321
    Time step :  322
    Time step :  323
    Time step :  324
    Time step :  325
    Time step :  326
    Time step :  327
    Time step :  328
    Time step :  329
    Time step :  330
    Time step :  331
    Time step :  332
    Time step :  333
    Time step :  334
    Time step :  335
    Time step :  336
    Time step :  337
    Time step :  338
    Time step :  339
    Time step :  340
    Time step :  341
    Time step :  342
    Time step :  343
    Time step :  344
    Time step :  345
    Time step :  346
    Time step :  347
    Time step :  348
    Time step :  349
    Time step :  350
    Time step :  351
    Time step :  352
    Time step :  353
    Time step :  354
    Time step :  355
    Time step :  356
    Time step :  357
    Time step :  358
    Time step :  359
    Time step :  360
    Time step :  361
    Time step :  362
    Time step :  363
    Time step :  364
    Time step :  365
    Time step :  366
    Time step :  367
    Time step :  368
    Time step :  369
    Time step :  370
    Time step :  371
    Time step :  372
    Time step :  373
    Time step :  374
    Time step :  375
    Time step :  376
    Time step :  377
    Time step :  378
    Time step :  379
    Time step :  380
    Time step :  381
    Time step :  382
    Time step :  383
    Time step :  384
    Time step :  385
    Time step :  386
    Time step :  387
    Time step :  388
    Time step :  389
    Time step :  390
    Time step :  391
    Time step :  392
    Time step :  393
    Time step :  394
    Time step :  395
    Time step :  396
    Time step :  397
    Time step :  398
    Time step :  399
    Time step :  400
    Time step :  401
    Time step :  402
    Time step :  403
    Time step :  404
    Time step :  405
    Time step :  406
    Time step :  407
    Time step :  408
    Time step :  409
    Time step :  410
    Time step :  411
    Time step :  412
    Time step :  413
    Time step :  414
    Time step :  415
    Time step :  416
    Time step :  417
    Time step :  418
    Time step :  419
    Time step :  420
    Time step :  421
    Time step :  422
    Time step :  423
    Time step :  424
    Time step :  425
    Time step :  426
    Time step :  427
    Time step :  428
    Time step :  429
    Time step :  430
    Time step :  431
    Time step :  432
    Time step :  433
    Time step :  434
    Time step :  435
    Time step :  436
    Time step :  437
    Time step :  438
    Time step :  439
    Time step :  440
    Time step :  441
    Time step :  442
    Time step :  443
    Time step :  444
    Time step :  445
    Time step :  446
    Time step :  447
    Time step :  448
    Time step :  449
    Time step :  450
    Time step :  451
    Time step :  452
    Time step :  453
    Time step :  454
    Time step :  455
    Time step :  456
    Time step :  457
    Time step :  458
    Time step :  459
    Time step :  460
    Time step :  461
    Time step :  462
    Time step :  463
    Time step :  464
    Time step :  465
    Time step :  466
    Time step :  467
    Time step :  468
    Time step :  469
    Time step :  470
    Time step :  471
    Time step :  472
    Time step :  473
    Time step :  474
    Time step :  475
    Time step :  476
    Time step :  477
    Time step :  478
    Time step :  479
    Time step :  480
    Time step :  481
    Time step :  482
    Time step :  483
    Time step :  484
    Time step :  485
    Time step :  486
    Time step :  487
    Time step :  488
    Time step :  489
    Time step :  490
    Time step :  491
    Time step :  492
    Time step :  493
    Time step :  494
    Time step :  495
    Time step :  496
    Time step :  497
    Time step :  498
    Time step :  499
    Time step :  500
    Time step :  501
    Time step :  502
    Time step :  503
    Time step :  504
    Time step :  505
    Time step :  506
    Time step :  507
    Time step :  508
    Time step :  509
    Time step :  510
    Time step :  511
    Time step :  512
    Time step :  513
    Time step :  514
    Time step :  515
    Time step :  516
    Time step :  517
    Time step :  518
    Time step :  519
    Time step :  520
    Time step :  521
    Time step :  522
    Time step :  523
    Time step :  524
    Time step :  525
    Time step :  526
    Time step :  527
    Time step :  528
    Time step :  529
    Time step :  530
    Time step :  531
    Time step :  532
    Time step :  533
    Time step :  534
    Time step :  535
    Time step :  536
    Time step :  537
    Time step :  538
    Time step :  539
    Time step :  540
    Time step :  541
    Time step :  542
    Time step :  543
    Time step :  544
    Time step :  545
    Time step :  546
    Time step :  547
    Time step :  548
    Time step :  549
    Time step :  550
    Time step :  551
    Time step :  552
    Time step :  553
    Time step :  554
    Time step :  555
    Time step :  556
    Time step :  557
    Time step :  558
    Time step :  559
    Time step :  560
    Time step :  561
    Time step :  562
    Time step :  563
    Time step :  564
    Time step :  565
    Time step :  566
    Time step :  567
    Time step :  568
    Time step :  569
    Time step :  570
    Time step :  571
    Time step :  572
    Time step :  573
    Time step :  574
    Time step :  575
    Time step :  576
    Time step :  577
    Time step :  578
    Time step :  579
    Time step :  580
    Time step :  581
    Time step :  582
    Time step :  583
    Time step :  584
    Time step :  585
    Time step :  586
    Time step :  587
    Time step :  588
    Time step :  589
    Time step :  590
    Time step :  591
    Time step :  592
    Time step :  593
    Time step :  594
    Time step :  595
    Time step :  596
    Time step :  597
    Time step :  598
    Time step :  599
    Time step :  600
    Time step :  601
    Time step :  602
    Time step :  603
    Time step :  604
    Time step :  605
    Time step :  606
    Time step :  607
    Time step :  608
    Time step :  609
    Time step :  610
    Time step :  611
    Time step :  612
    Time step :  613
    Time step :  614
    Time step :  615
    Time step :  616
    Time step :  617
    Time step :  618
    Time step :  619
    Time step :  620
    Time step :  621
    Time step :  622
    Time step :  623
    Time step :  624
    Time step :  625
    Time step :  626
    Time step :  627
    Time step :  628
    Time step :  629
    Time step :  630
    Time step :  631
    Time step :  632
    Time step :  633
    Time step :  634
    Time step :  635
    Time step :  636
    Time step :  637
    Time step :  638
    Time step :  639
    Time step :  640
    Time step :  641
    Time step :  642
    Time step :  643
    Time step :  644
    Time step :  645
    Time step :  646
    Time step :  647
    Time step :  648
    Time step :  649
    Time step :  650
    Time step :  651
    Time step :  652
    Time step :  653
    Time step :  654
    Time step :  655
    Time step :  656
    Time step :  657
    Time step :  658
    Time step :  659
    Time step :  660
    Time step :  661
    Time step :  662
    Time step :  663
    Time step :  664
    Time step :  665
    Time step :  666
    Time step :  667
    Time step :  668
    Time step :  669
    Time step :  670
    Time step :  671
    Time step :  672
    Time step :  673
    Time step :  674
    Time step :  675
    Time step :  676
    Time step :  677
    Time step :  678
    Time step :  679
    Time step :  680
    Time step :  681
    Time step :  682
    Time step :  683
    Time step :  684
    Time step :  685
    Time step :  686
    Time step :  687
    Time step :  688
    Time step :  689
    Time step :  690
    Time step :  691
    Time step :  692
    Time step :  693
    Time step :  694
    Time step :  695
    Time step :  696
    Time step :  697
    Time step :  698
    Time step :  699
    Time step :  700
    Time step :  701
    Time step :  702
    Time step :  703
    Time step :  704
    Time step :  705
    Time step :  706
    Time step :  707
    Time step :  708
    Time step :  709
    Time step :  710
    Time step :  711
    Time step :  712
    Time step :  713
    Time step :  714
    Time step :  715
    Time step :  716
    Time step :  717
    Time step :  718
    Time step :  719
    Time step :  720
    Time step :  721
    Time step :  722
    Time step :  723
    Time step :  724
    Time step :  725
    Time step :  726
    Time step :  727
    Time step :  728
    Time step :  729
    Time step :  730
    Time step :  731
    Time step :  732
    Time step :  733
    Time step :  734
    Time step :  735
    Time step :  736
    Time step :  737
    Time step :  738
    Time step :  739
    Time step :  740
    Time step :  741
    Time step :  742
    Time step :  743
    Time step :  744
    Time step :  745
    Time step :  746
    Time step :  747
    Time step :  748
    Time step :  749
    Time step :  750
    Time step :  751
    Time step :  752
    Time step :  753
    Time step :  754
    Time step :  755
    Time step :  756
    Time step :  757
    Time step :  758
    Time step :  759
    Time step :  760
    Time step :  761
    Time step :  762
    Time step :  763
    Time step :  764
    Time step :  765
    Time step :  766
    Time step :  767
    Time step :  768
    Time step :  769
    Time step :  770
    Time step :  771
    Time step :  772
    Time step :  773
    Time step :  774
    Time step :  775
    Time step :  776
    Time step :  777
    Time step :  778
    Time step :  779
    Time step :  780
    Time step :  781
    Time step :  782
    Time step :  783
    Time step :  784
    Time step :  785
    Time step :  786
    Time step :  787
    Time step :  788
    Time step :  789
    Time step :  790
    Time step :  791
    Time step :  792
    Time step :  793
    Time step :  794
    Time step :  795
    Time step :  796
    Time step :  797
    Time step :  798
    Time step :  799
    Time step :  800
    Time step :  801
    Time step :  802
    Time step :  803
    Time step :  804
    Time step :  805
    Time step :  806
    Time step :  807
    Time step :  808
    Time step :  809
    Time step :  810
    Time step :  811
    Time step :  812
    Time step :  813
    Time step :  814
    Time step :  815
    Time step :  816
    Time step :  817
    Time step :  818
    Time step :  819
    Time step :  820
    Time step :  821
    Time step :  822
    Time step :  823
    Time step :  824
    Time step :  825
    Time step :  826
    Time step :  827
    Time step :  828
    Time step :  829
    Time step :  830
    Time step :  831
    Time step :  832
    Time step :  833
    Time step :  834
    Time step :  835
    Time step :  836
    Time step :  837
    Time step :  838
    Time step :  839
    Time step :  840
    Time step :  841
    Time step :  842
    Time step :  843
    Time step :  844
    Time step :  845
    Time step :  846
    Time step :  847
    Time step :  848
    Time step :  849
    Time step :  850
    Time step :  851
    Time step :  852
    Time step :  853
    Time step :  854
    Time step :  855
    Time step :  856
    Time step :  857
    Time step :  858
    Time step :  859
    Time step :  860
    Time step :  861
    Time step :  862
    Time step :  863
    Time step :  864
    Time step :  865
    Time step :  866
    Time step :  867
    Time step :  868
    Time step :  869
    Time step :  870
    Time step :  871
    Time step :  872
    Time step :  873
    Time step :  874
    Time step :  875
    Time step :  876
    Time step :  877
    Time step :  878
    Time step :  879
    Time step :  880
    Time step :  881
    Time step :  882
    Time step :  883
    Time step :  884
    Time step :  885
    Time step :  886
    Time step :  887
    Time step :  888
    Time step :  889
    Time step :  890
    Time step :  891
    Time step :  892
    Time step :  893
    Time step :  894
    Time step :  895
    Time step :  896
    Time step :  897
    Time step :  898
    Time step :  899
    Time step :  900
    Time step :  901
    Time step :  902
    Time step :  903
    Time step :  904
    Time step :  905
    Time step :  906
    Time step :  907
    Time step :  908
    Time step :  909
    Time step :  910
    Time step :  911
    Time step :  912
    Time step :  913
    Time step :  914
    Time step :  915
    Time step :  916
    Time step :  917
    Time step :  918
    Time step :  919
    Time step :  920
    Time step :  921
    Time step :  922
    Time step :  923
    Time step :  924
    Time step :  925
    Time step :  926
    Time step :  927
    Time step :  928
    Time step :  929
    Time step :  930
    Time step :  931
    Time step :  932
    Time step :  933
    Time step :  934
    Time step :  935
    Time step :  936
    Time step :  937
    Time step :  938
    Time step :  939
    Time step :  940
    Time step :  941
    Time step :  942
    Time step :  943
    Time step :  944
    Time step :  945
    Time step :  946
    Time step :  947
    Time step :  948
    Time step :  949
    Time step :  950
    Time step :  951
    Time step :  952
    Time step :  953
    Time step :  954
    Time step :  955
    Time step :  956
    Time step :  957
    Time step :  958
    Time step :  959
    Time step :  960
    Time step :  961
    Time step :  962
    Time step :  963
    Time step :  964
    Time step :  965
    Time step :  966
    Time step :  967
    Time step :  968
    Time step :  969
    Time step :  970
    Time step :  971
    Time step :  972
    Time step :  973
    Time step :  974
    Time step :  975
    Time step :  976
    Time step :  977
    Time step :  978
    Time step :  979
    Time step :  980
    Time step :  981
    Time step :  982
    Time step :  983
    Time step :  984
    Time step :  985
    Time step :  986
    Time step :  987
    Time step :  988
    Time step :  989
    Time step :  990
    Time step :  991
    Time step :  992
    Time step :  993
    Time step :  994
    Time step :  995
    Time step :  996
    Time step :  997
    Time step :  998
    Time step :  999


##### A note on units
<div class="alert alert-block alert-warning">
So far we have been rather loose with the units. For example, we set $L=50$ without specifying what "50" means in terms of real physical units. 

We actually work in the "simulation units", i.e., all numbers as quoted appear in the implementation of the force and torque laws as well as in the equations of motion, i.e. no parameters are rescaled. In other words, from the implementation point of view, all equations given above have been non-dimensionalized. It is up the the user to give the physical interpretation to the values of parameter used.

For example, we can work in the system of units where radius of a particle sets the unit of length. In this case, we would have $L=50a$. Similarly, we could have set the unit of time $t^*=\frac{\gamma_t}{k}$, in which case the time step $\delta t = 0.01t^*$ and the $\frac{\gamma_r}{J}$ would have to be expressed in terms of $t^*$, etc.

To put it rather mundanely, the computer does not care if the units are correct and what they mean. It is up to the simulator to assign physical meaning to what these parameters.
</div>

### Step by step guide through the working example

Let us now dissect the example above line by line and look into what is going on under the hood.  

In order to use the pymd package we need to load it. In particular, we import the md module.



```python
from pymd.md import *
```

The source code on the entire module can be found in the *ABPtutotial/Python* directory. 

#### Creating the System object

In the following line, we crate an instance of the *System* class. The *System* class stores the system we are simulating and the information about the simulation box. It is also in charge of reading the initial configuration from a JSON file and it also builds and updates the neighbor list. 


```python
s = System(rcut = 3.0, pad = 0.5)   # Create a system object with neighbour list cutoff rcut = 3.0 and padding distance 0.5
s.read_init('init.json')            # Read in the initial configuration
```

In our example, the instance of the *System* class is created with the neighbor list cutoff set to $r_{cut}=3.0$ and padding distance $d_{pad}=0.5$. For the system parameters used below, this corresponds approximately to 10 steps between neighbor list rebuilds.

After the *System* object has been created, we call the *read_init* member function to read in the size of the simulation box and the initial location and direction of the polarity vector for each particle.

*System* class is a part of the pymd.md core.

##### A note on neighbor lists
<div class="alert alert-block alert-info">
Neighbor list is one of the central parts of almost every particle-based code and yet is has almost nothing to do with physics. It is, however, a very handy tool to speed up calculations of forces and torques. Namely, with the exception of dealing with charged systems, most particle-particle interactions as pairwise and short-range. In other words, each particle interacts only with a handful of its immediate neighbors. In the example we study here, each particle interacts only with particles it overlaps with. 
    
The most time consuming part of the simulation is the computation of forces acting on each particle. This computation has to be performed at each time step. At the face value, it is an $O(N^2)$ problem as for every particle one has to loop over the entire system. One can half the time by using the third Newton's law. However, in practice, this is usually not enough. For example, for a modest system with $10^3$ particles, the "force loop" would have to iterate $\sim10^6$ times. 
    
For a system with short range interactions it is immediately clear that the vast majority of terms in this double loop will be equal to zero. Therefore, it is highly beneficial to keep track of who the neighbors of each particle are and only include them in the calculation. The performance gain of this approach is hard to exaggerate, however, it comes at the cost of significant increase in code complexity, i.e., neighbor lists are not easy to implement.
</div>

<div class="alert alert-block alert-info">
The idea is, however, simple. We keep a list of neighbors for each particle. All neighbors are within some cutoff distance (typically comparable to the interaction range). This cutoff is extended by a padding distance. If any of the particles moves more than half of the padding distance there is no longer guarantee that the current neighbor list represents the true state of the system and it has to be rebuild. Typically, one chooses the padding such that the neighbor list is rebuilt once every 10-20 time steps. This often requires some experimenting with parameters.
    
In this tutorial, we use so-called cell lists to improve the speed of neighbor list rebuilds. This approach is efficient for dense systems. For systems with low density, it is more optimal to use octtrees (or quadtrees in 2d). However, this is beyond the scope of this tutorial.
</div>

Below is a visual representation of a neighbor list:
<div align="center">
    
<img src="./nlist.png" style="width: 600px;"/>
</div>

#### The Evolver class

Now that we have created the *System* object and populated it with the initial configuration, we proceed to create the *Evolver* object:


```python
e = Evolver(s)                      # Creates an evolver object
```

The *Evolver* class is the workhorse of our toy simulation. It's central member function is *evolve* which performs the one step of the simulation. Its task is to:

1. Ensure that the neighbor list is up to date;
2. Perform the integration pre-step;
3. Compute all forces and torques on each particle;
4. Perform the integration post-step;
5. Apply periodic boundary conditions.

In our design, one can have multiple force and torque laws apply in the system. In addition, multiple integrators can act in the same system. More about what this means below.

*Evolver* class is a part of the pymd.md core.

#### The Dump class

The *Dump* class handles output of the simulation run. This is the most important part of the code for interacting with the postprocessing analysis tools. It takes snapshots of the state of the system at a given instance in time and saves it to a file of a certain type. Here, we support outputs in JSON and VTP formats.

*Dump* class is a part of the pymd.md core.


```python
d = Dump(s)                         # Create a dump object
```

#### Forces and torques

In our implementation, each non-stochastic term in equations of motion is implemented as a separate class. Terms in the equation of motion for particle's position are all given a generic name *force* while terms that contribute to the orientation of particle's polarity are called *torque*. Note particles in our model are disks and that rotating a disk will not stir the system, so the term *torque* is a bit loose. 

All *forces* and *torques* are stored in two lists in the *Evolver* class. The *evolve* member function of the *Evolver* class ensured that in each time step forces and torques on every particle are set to zero, before they are recomputed term by term.

While this design introduces some performance overhead (e.g., distances between particles have to be compute multiple times in the same time step), it provides great level flexibility since adding an additional term to the equations of motion requires just implementing a new *Force* class and adding it to the *Evolver*'s list of forces.

These classes are defined in the *forces* and *torques* directories. 


```python
hf = HarmonicForce(s, 10.0, 2.0)    # Create pairwise repulsive interactions with spring contant k = 10 and range 2.0
sp = SelfPropulsion(s, 1.0)         # Create self-propulsion, self-propulsion strength alpha = 1.0
pa = PolarAlign(s, 1.0, 2.0)        # Create pairwise polar alignment with alignment strength J = 1.0 and range 2.0

# Register all forces, torques and integrators with the evolver object
e.add_force(hf)                    
e.add_force(sp)
e.add_torque(pa)
```

#### Integrators

Finally, the equations of motion are solved numerically using a discretization scheme. This is done by a set of classes that we call *Integrator* classes. Each of these classes implement one of the standard discretization schemed for solving a set of coupled ODEs. In the spirit of our modular design, solver of each equation of motion is implemented as a separate *Integrator* class allowing the code to be modular and simple to maintain.

In our example, both equations of motion are overdamped and not stiff. However, both equations of motion contain stochastic terms. Therefore, in both cases we implement the simple first-order Euler-Maruyama scheme.

For the equation of motion for particle's position with time step $\delta t$, the discretization scheme (e.g., for the $x-$component of the position of particle $i$) is:

$x_i(t+\delta t) = x_i(t) + \delta t\left(v_0 n_{i,x} - \frac{1}{\gamma_t}\sum_j k\left(2a - r_{ij}\right)\frac{x_i}{r_{ij}} \right) + \sqrt{2\frac{T}{\gamma_r}\delta t}\mathcal{N}(0,1)$, 

where $\mathcal{N}(0,1)$ is a random number drawn from a Gaussian distribution with the zero mean and unit variance. This integrator is implemented in the *BrownianIntegrator* class. 

The implementation for the direction of the particle's orientation is similar and is implemented in the *BrownianRotIntegrator* class. Note that, for convenience, in this class we also introduced a parameter $T$, such that the rotation diffusion coefficient is defined as $D_r = \frac{T}{\gamma_r}$. 

##### A note on numerical integration of equation of motion
<div class="alert alert-block alert-info">

Discretization of a set of ordinary equations of motion has be a field of active research in applied mathematics. Choosing the right method depends on the properties of the system of equations of interest and can often be very involved, especially if one requires a very accurate method (e.g., in studies of planetary dynamics). In the soft and active matter simulations, in most cases, using simple first- and second-order discretization schemes is usually sufficient. Since we are interested in collective behavior of a large number of particles and those particles scatter off of each other frequently, the precise knowledge of the trajectory of each particle is not essential.
    
In the case where inertial terms are not neglected, equations of motion are of the second order and a solving them with a simple first-order Euler method is usually not sufficient. However, the standard second order integrator, such as the half-step velocity-Verlet method is typically sufficient. The key difference between first-order integrators used in this example and a velocity-Verlet scheme is that the later involves a "pre-step" (i.e., a half-step of length $\frac{\delta t}{2}$) before forces and torques are computed. In order to make implementing such integrators in our code, each integrator class defines *prestep* and *poststep* member function. *prestep* (*poststep*) is invoked before (after) forces and and torques are computed.
</div>

<div class="alert alert-block alert-info">
Two particularly pedagogical texts on the theory behind numerical integration of equations of motion are:

1. Benedict Leimkuhler and Sebastian Reich, Simulating Hamiltonian Dynamics, Cambridge University Press, 2005. 
2. Ben Leimkuhler and Charles Matthews, Molecular Dynamics: With Deterministic and Stochastic Numerical Methods, Springer; 2015 edition.
</div>


```python
pos_integ = BrownianIntegrator(s, T = 0.0, gamma = 1.0)       # Integrator for updating particle position, friction gamma = 1.0 and no thermal noise
rot_integ = BrownianRotIntegrator(s, T = 0.1, gamma = 1.0)    # Integrator for updating particle oriantation, friction gamma = 1.0, "rotation" T = 0.1, D_r = 0.0 

# add the two integrtors to the Evolver's list of integrtors
e.add_integrator(pos_integ)
e.add_integrator(rot_integ)
```

#### Iterating over time

Now that we have all ingredients in place, we can run a simulation. In our implementation, this is just a simple "for" loop over the predefined set of time steps. Inside this loop we perform the basic data collection, e.g., saving snapshots of the state of the system.


```python
dt = 0.01    # Simulation time step
# Run simulation for 1,000 time steps (total simulation time is 10 time units)
for t in range(1000):
  print("Time step : ", t)
  e.evolve(dt)       # Evolve the system by one time step of length dt
  if t % 10 == 0:    # Produce snapshot of the simulation once every 10 time steps
    d.dump_vtp('test_{:05d}.vtp'.format(t))
```

    Time step :  0
    Time step :  1
    Time step :  2
    Time step :  3
    Time step :  4
    Time step :  5
    Time step :  6
    Time step :  7
    Time step :  8
    Time step :  9
    Time step :  10
    Time step :  11
    Time step :  12
    Time step :  13
    Time step :  14
    Time step :  15
    Time step :  16
    Time step :  17
    Time step :  18
    Time step :  19
    Time step :  20
    Time step :  21
    Time step :  22
    Time step :  23
    Time step :  24
    Time step :  25
    Time step :  26
    Time step :  27
    Time step :  28
    Time step :  29
    Time step :  30
    Time step :  31
    Time step :  32
    Time step :  33
    Time step :  34
    Time step :  35
    Time step :  36
    Time step :  37
    Time step :  38
    Time step :  39
    Time step :  40
    Time step :  41
    Time step :  42
    Time step :  43
    Time step :  44
    Time step :  45
    Time step :  46
    Time step :  47
    Time step :  48
    Time step :  49
    Time step :  50
    Time step :  51
    Time step :  52
    Time step :  53
    Time step :  54
    Time step :  55
    Time step :  56
    Time step :  57
    Time step :  58
    Time step :  59
    Time step :  60
    Time step :  61
    Time step :  62
    Time step :  63
    Time step :  64
    Time step :  65
    Time step :  66
    Time step :  67
    Time step :  68
    Time step :  69
    Time step :  70
    Time step :  71
    Time step :  72
    Time step :  73
    Time step :  74
    Time step :  75
    Time step :  76
    Time step :  77
    Time step :  78
    Time step :  79
    Time step :  80
    Time step :  81
    Time step :  82
    Time step :  83
    Time step :  84
    Time step :  85
    Time step :  86
    Time step :  87
    Time step :  88
    Time step :  89
    Time step :  90
    Time step :  91
    Time step :  92
    Time step :  93
    Time step :  94
    Time step :  95
    Time step :  96
    Time step :  97
    Time step :  98
    Time step :  99
    Time step :  100
    Time step :  101
    Time step :  102
    Time step :  103
    Time step :  104
    Time step :  105
    Time step :  106
    Time step :  107
    Time step :  108
    Time step :  109
    Time step :  110
    Time step :  111
    Time step :  112
    Time step :  113
    Time step :  114
    Time step :  115
    Time step :  116
    Time step :  117
    Time step :  118
    Time step :  119
    Time step :  120
    Time step :  121
    Time step :  122
    Time step :  123
    Time step :  124
    Time step :  125
    Time step :  126
    Time step :  127
    Time step :  128
    Time step :  129
    Time step :  130
    Time step :  131
    Time step :  132
    Time step :  133
    Time step :  134
    Time step :  135
    Time step :  136
    Time step :  137
    Time step :  138
    Time step :  139
    Time step :  140
    Time step :  141
    Time step :  142
    Time step :  143
    Time step :  144
    Time step :  145
    Time step :  146
    Time step :  147
    Time step :  148
    Time step :  149
    Time step :  150
    Time step :  151
    Time step :  152
    Time step :  153
    Time step :  154
    Time step :  155
    Time step :  156
    Time step :  157
    Time step :  158
    Time step :  159
    Time step :  160
    Time step :  161
    Time step :  162
    Time step :  163
    Time step :  164
    Time step :  165
    Time step :  166
    Time step :  167
    Time step :  168
    Time step :  169
    Time step :  170
    Time step :  171
    Time step :  172
    Time step :  173
    Time step :  174
    Time step :  175
    Time step :  176
    Time step :  177
    Time step :  178
    Time step :  179
    Time step :  180
    Time step :  181
    Time step :  182
    Time step :  183
    Time step :  184
    Time step :  185
    Time step :  186
    Time step :  187
    Time step :  188
    Time step :  189
    Time step :  190
    Time step :  191
    Time step :  192
    Time step :  193
    Time step :  194
    Time step :  195
    Time step :  196
    Time step :  197
    Time step :  198
    Time step :  199
    Time step :  200
    Time step :  201
    Time step :  202
    Time step :  203
    Time step :  204
    Time step :  205
    Time step :  206
    Time step :  207
    Time step :  208
    Time step :  209
    Time step :  210
    Time step :  211
    Time step :  212
    Time step :  213
    Time step :  214
    Time step :  215
    Time step :  216
    Time step :  217
    Time step :  218
    Time step :  219
    Time step :  220
    Time step :  221
    Time step :  222
    Time step :  223
    Time step :  224
    Time step :  225
    Time step :  226
    Time step :  227
    Time step :  228
    Time step :  229
    Time step :  230
    Time step :  231
    Time step :  232
    Time step :  233
    Time step :  234
    Time step :  235
    Time step :  236
    Time step :  237
    Time step :  238
    Time step :  239
    Time step :  240
    Time step :  241
    Time step :  242
    Time step :  243
    Time step :  244
    Time step :  245
    Time step :  246
    Time step :  247
    Time step :  248
    Time step :  249
    Time step :  250
    Time step :  251
    Time step :  252
    Time step :  253
    Time step :  254
    Time step :  255
    Time step :  256
    Time step :  257
    Time step :  258
    Time step :  259
    Time step :  260
    Time step :  261
    Time step :  262
    Time step :  263
    Time step :  264
    Time step :  265
    Time step :  266
    Time step :  267
    Time step :  268
    Time step :  269
    Time step :  270
    Time step :  271
    Time step :  272
    Time step :  273
    Time step :  274
    Time step :  275
    Time step :  276
    Time step :  277
    Time step :  278
    Time step :  279
    Time step :  280
    Time step :  281
    Time step :  282
    Time step :  283
    Time step :  284
    Time step :  285
    Time step :  286
    Time step :  287
    Time step :  288
    Time step :  289
    Time step :  290
    Time step :  291
    Time step :  292
    Time step :  293
    Time step :  294
    Time step :  295
    Time step :  296
    Time step :  297
    Time step :  298
    Time step :  299
    Time step :  300
    Time step :  301
    Time step :  302
    Time step :  303
    Time step :  304
    Time step :  305
    Time step :  306
    Time step :  307
    Time step :  308
    Time step :  309
    Time step :  310
    Time step :  311
    Time step :  312
    Time step :  313
    Time step :  314
    Time step :  315
    Time step :  316
    Time step :  317
    Time step :  318
    Time step :  319
    Time step :  320
    Time step :  321
    Time step :  322
    Time step :  323
    Time step :  324
    Time step :  325
    Time step :  326
    Time step :  327
    Time step :  328
    Time step :  329
    Time step :  330
    Time step :  331
    Time step :  332
    Time step :  333
    Time step :  334
    Time step :  335
    Time step :  336
    Time step :  337
    Time step :  338
    Time step :  339
    Time step :  340
    Time step :  341
    Time step :  342
    Time step :  343
    Time step :  344
    Time step :  345
    Time step :  346
    Time step :  347
    Time step :  348
    Time step :  349
    Time step :  350
    Time step :  351
    Time step :  352
    Time step :  353
    Time step :  354
    Time step :  355
    Time step :  356
    Time step :  357
    Time step :  358
    Time step :  359
    Time step :  360
    Time step :  361
    Time step :  362
    Time step :  363
    Time step :  364
    Time step :  365
    Time step :  366
    Time step :  367
    Time step :  368
    Time step :  369
    Time step :  370
    Time step :  371
    Time step :  372
    Time step :  373
    Time step :  374
    Time step :  375
    Time step :  376
    Time step :  377
    Time step :  378
    Time step :  379
    Time step :  380
    Time step :  381
    Time step :  382
    Time step :  383
    Time step :  384
    Time step :  385
    Time step :  386
    Time step :  387
    Time step :  388
    Time step :  389
    Time step :  390
    Time step :  391
    Time step :  392
    Time step :  393
    Time step :  394
    Time step :  395
    Time step :  396
    Time step :  397
    Time step :  398
    Time step :  399
    Time step :  400
    Time step :  401
    Time step :  402
    Time step :  403
    Time step :  404
    Time step :  405
    Time step :  406
    Time step :  407
    Time step :  408
    Time step :  409
    Time step :  410
    Time step :  411
    Time step :  412
    Time step :  413
    Time step :  414
    Time step :  415
    Time step :  416
    Time step :  417
    Time step :  418
    Time step :  419
    Time step :  420
    Time step :  421
    Time step :  422
    Time step :  423
    Time step :  424
    Time step :  425
    Time step :  426
    Time step :  427
    Time step :  428
    Time step :  429
    Time step :  430
    Time step :  431
    Time step :  432
    Time step :  433
    Time step :  434
    Time step :  435
    Time step :  436
    Time step :  437
    Time step :  438
    Time step :  439
    Time step :  440
    Time step :  441
    Time step :  442
    Time step :  443
    Time step :  444
    Time step :  445
    Time step :  446
    Time step :  447
    Time step :  448
    Time step :  449
    Time step :  450
    Time step :  451
    Time step :  452
    Time step :  453
    Time step :  454
    Time step :  455
    Time step :  456
    Time step :  457
    Time step :  458
    Time step :  459
    Time step :  460
    Time step :  461
    Time step :  462
    Time step :  463
    Time step :  464
    Time step :  465
    Time step :  466
    Time step :  467
    Time step :  468
    Time step :  469
    Time step :  470
    Time step :  471
    Time step :  472
    Time step :  473
    Time step :  474
    Time step :  475
    Time step :  476
    Time step :  477
    Time step :  478
    Time step :  479
    Time step :  480
    Time step :  481
    Time step :  482
    Time step :  483
    Time step :  484
    Time step :  485
    Time step :  486
    Time step :  487
    Time step :  488
    Time step :  489
    Time step :  490
    Time step :  491
    Time step :  492
    Time step :  493
    Time step :  494
    Time step :  495
    Time step :  496
    Time step :  497
    Time step :  498
    Time step :  499
    Time step :  500
    Time step :  501
    Time step :  502
    Time step :  503
    Time step :  504
    Time step :  505
    Time step :  506
    Time step :  507
    Time step :  508
    Time step :  509
    Time step :  510
    Time step :  511
    Time step :  512
    Time step :  513
    Time step :  514
    Time step :  515
    Time step :  516
    Time step :  517
    Time step :  518
    Time step :  519
    Time step :  520
    Time step :  521
    Time step :  522
    Time step :  523
    Time step :  524
    Time step :  525
    Time step :  526
    Time step :  527
    Time step :  528
    Time step :  529
    Time step :  530
    Time step :  531
    Time step :  532
    Time step :  533
    Time step :  534
    Time step :  535
    Time step :  536
    Time step :  537
    Time step :  538
    Time step :  539
    Time step :  540
    Time step :  541
    Time step :  542
    Time step :  543
    Time step :  544
    Time step :  545
    Time step :  546
    Time step :  547
    Time step :  548
    Time step :  549
    Time step :  550
    Time step :  551
    Time step :  552
    Time step :  553
    Time step :  554
    Time step :  555
    Time step :  556
    Time step :  557
    Time step :  558
    Time step :  559
    Time step :  560
    Time step :  561
    Time step :  562
    Time step :  563
    Time step :  564
    Time step :  565
    Time step :  566
    Time step :  567
    Time step :  568
    Time step :  569
    Time step :  570
    Time step :  571
    Time step :  572
    Time step :  573
    Time step :  574
    Time step :  575
    Time step :  576
    Time step :  577
    Time step :  578
    Time step :  579
    Time step :  580
    Time step :  581
    Time step :  582
    Time step :  583
    Time step :  584
    Time step :  585
    Time step :  586
    Time step :  587
    Time step :  588
    Time step :  589
    Time step :  590
    Time step :  591
    Time step :  592
    Time step :  593
    Time step :  594
    Time step :  595
    Time step :  596
    Time step :  597
    Time step :  598
    Time step :  599
    Time step :  600
    Time step :  601
    Time step :  602
    Time step :  603
    Time step :  604
    Time step :  605
    Time step :  606
    Time step :  607
    Time step :  608
    Time step :  609
    Time step :  610
    Time step :  611
    Time step :  612
    Time step :  613
    Time step :  614
    Time step :  615
    Time step :  616
    Time step :  617
    Time step :  618
    Time step :  619
    Time step :  620
    Time step :  621
    Time step :  622
    Time step :  623
    Time step :  624
    Time step :  625
    Time step :  626
    Time step :  627
    Time step :  628
    Time step :  629
    Time step :  630
    Time step :  631
    Time step :  632
    Time step :  633
    Time step :  634
    Time step :  635
    Time step :  636
    Time step :  637
    Time step :  638
    Time step :  639
    Time step :  640
    Time step :  641
    Time step :  642
    Time step :  643
    Time step :  644
    Time step :  645
    Time step :  646
    Time step :  647
    Time step :  648
    Time step :  649
    Time step :  650
    Time step :  651
    Time step :  652
    Time step :  653
    Time step :  654
    Time step :  655
    Time step :  656
    Time step :  657
    Time step :  658
    Time step :  659
    Time step :  660
    Time step :  661
    Time step :  662
    Time step :  663
    Time step :  664
    Time step :  665
    Time step :  666
    Time step :  667
    Time step :  668
    Time step :  669
    Time step :  670
    Time step :  671
    Time step :  672
    Time step :  673
    Time step :  674
    Time step :  675
    Time step :  676
    Time step :  677
    Time step :  678
    Time step :  679
    Time step :  680
    Time step :  681
    Time step :  682
    Time step :  683
    Time step :  684
    Time step :  685
    Time step :  686
    Time step :  687
    Time step :  688
    Time step :  689
    Time step :  690
    Time step :  691
    Time step :  692
    Time step :  693
    Time step :  694
    Time step :  695
    Time step :  696
    Time step :  697
    Time step :  698
    Time step :  699
    Time step :  700
    Time step :  701
    Time step :  702
    Time step :  703
    Time step :  704
    Time step :  705
    Time step :  706
    Time step :  707
    Time step :  708
    Time step :  709
    Time step :  710
    Time step :  711
    Time step :  712
    Time step :  713
    Time step :  714
    Time step :  715
    Time step :  716
    Time step :  717
    Time step :  718
    Time step :  719
    Time step :  720
    Time step :  721
    Time step :  722
    Time step :  723
    Time step :  724
    Time step :  725
    Time step :  726
    Time step :  727
    Time step :  728
    Time step :  729
    Time step :  730
    Time step :  731
    Time step :  732
    Time step :  733
    Time step :  734
    Time step :  735
    Time step :  736
    Time step :  737
    Time step :  738
    Time step :  739
    Time step :  740
    Time step :  741
    Time step :  742
    Time step :  743
    Time step :  744
    Time step :  745
    Time step :  746
    Time step :  747
    Time step :  748
    Time step :  749
    Time step :  750
    Time step :  751
    Time step :  752
    Time step :  753
    Time step :  754
    Time step :  755
    Time step :  756
    Time step :  757
    Time step :  758
    Time step :  759
    Time step :  760
    Time step :  761
    Time step :  762
    Time step :  763
    Time step :  764
    Time step :  765
    Time step :  766
    Time step :  767
    Time step :  768
    Time step :  769
    Time step :  770
    Time step :  771
    Time step :  772
    Time step :  773
    Time step :  774
    Time step :  775
    Time step :  776
    Time step :  777
    Time step :  778
    Time step :  779
    Time step :  780
    Time step :  781
    Time step :  782
    Time step :  783
    Time step :  784
    Time step :  785
    Time step :  786
    Time step :  787
    Time step :  788
    Time step :  789
    Time step :  790
    Time step :  791
    Time step :  792
    Time step :  793
    Time step :  794
    Time step :  795
    Time step :  796
    Time step :  797
    Time step :  798
    Time step :  799
    Time step :  800
    Time step :  801
    Time step :  802
    Time step :  803
    Time step :  804
    Time step :  805
    Time step :  806
    Time step :  807
    Time step :  808
    Time step :  809
    Time step :  810
    Time step :  811
    Time step :  812
    Time step :  813
    Time step :  814
    Time step :  815
    Time step :  816
    Time step :  817
    Time step :  818
    Time step :  819
    Time step :  820
    Time step :  821
    Time step :  822
    Time step :  823
    Time step :  824
    Time step :  825
    Time step :  826
    Time step :  827
    Time step :  828
    Time step :  829
    Time step :  830
    Time step :  831
    Time step :  832
    Time step :  833
    Time step :  834
    Time step :  835
    Time step :  836
    Time step :  837
    Time step :  838
    Time step :  839
    Time step :  840
    Time step :  841
    Time step :  842
    Time step :  843
    Time step :  844
    Time step :  845
    Time step :  846
    Time step :  847
    Time step :  848
    Time step :  849
    Time step :  850
    Time step :  851
    Time step :  852
    Time step :  853
    Time step :  854
    Time step :  855
    Time step :  856
    Time step :  857
    Time step :  858
    Time step :  859
    Time step :  860
    Time step :  861
    Time step :  862
    Time step :  863
    Time step :  864
    Time step :  865
    Time step :  866
    Time step :  867
    Time step :  868
    Time step :  869
    Time step :  870
    Time step :  871
    Time step :  872
    Time step :  873
    Time step :  874
    Time step :  875
    Time step :  876
    Time step :  877
    Time step :  878
    Time step :  879
    Time step :  880
    Time step :  881
    Time step :  882
    Time step :  883
    Time step :  884
    Time step :  885
    Time step :  886
    Time step :  887
    Time step :  888
    Time step :  889
    Time step :  890
    Time step :  891
    Time step :  892
    Time step :  893
    Time step :  894
    Time step :  895
    Time step :  896
    Time step :  897
    Time step :  898
    Time step :  899
    Time step :  900
    Time step :  901
    Time step :  902
    Time step :  903
    Time step :  904
    Time step :  905
    Time step :  906
    Time step :  907
    Time step :  908
    Time step :  909
    Time step :  910
    Time step :  911
    Time step :  912
    Time step :  913
    Time step :  914
    Time step :  915
    Time step :  916
    Time step :  917
    Time step :  918
    Time step :  919
    Time step :  920
    Time step :  921
    Time step :  922
    Time step :  923
    Time step :  924
    Time step :  925
    Time step :  926
    Time step :  927
    Time step :  928
    Time step :  929
    Time step :  930
    Time step :  931
    Time step :  932
    Time step :  933
    Time step :  934
    Time step :  935
    Time step :  936
    Time step :  937
    Time step :  938
    Time step :  939
    Time step :  940
    Time step :  941
    Time step :  942
    Time step :  943
    Time step :  944
    Time step :  945
    Time step :  946
    Time step :  947
    Time step :  948
    Time step :  949
    Time step :  950
    Time step :  951
    Time step :  952
    Time step :  953
    Time step :  954
    Time step :  955
    Time step :  956
    Time step :  957
    Time step :  958
    Time step :  959
    Time step :  960
    Time step :  961
    Time step :  962
    Time step :  963
    Time step :  964
    Time step :  965
    Time step :  966
    Time step :  967
    Time step :  968
    Time step :  969
    Time step :  970
    Time step :  971
    Time step :  972
    Time step :  973
    Time step :  974
    Time step :  975
    Time step :  976
    Time step :  977
    Time step :  978
    Time step :  979
    Time step :  980
    Time step :  981
    Time step :  982
    Time step :  983
    Time step :  984
    Time step :  985
    Time step :  986
    Time step :  987
    Time step :  988
    Time step :  989
    Time step :  990
    Time step :  991
    Time step :  992
    Time step :  993
    Time step :  994
    Time step :  995
    Time step :  996
    Time step :  997
    Time step :  998
    Time step :  999


#### Final remarks on implementation 

In the case of a full production-grade code, it would be convenient to add several more components. For example:

*Logger* classes would be in charge of handling logging of various observables that can be compute on "fly", such as the total energy of the system (arguably, of not much use in an active, self-propelled system), pressure, stress, etc. 

*Event* classes would be reporting key events, such as reporting if the new *force* term was added, what is the frequency of saving the data, etc.

It would also be useful to save the entire state of the simulation (i.e., all parameters for each particle and all forces, torques and integrators) for future restarts.

One of the key parts of any good code, which is, unfortunately, often overlooked is having a set of **automatic tests**. As the code grows in complexity, making sure that adding new features does not break the expected behavior of the existing components becomes a challenging task. A modular design that decouples different parts of the code from each other as much as possible, together with a powerful set of automatic tests is essential to make this task tractable. 

### Visualizing results

There are many excellent tools for visualizing the results of a particle-based simulation. Commonly used are VMD, PyMol and Chimera, to name a few. In this tutorial we use Paraview.

### Analyzing results of a simulation

This is the third step in the simulation workflow. Once we have collected particle trajectories it is time to extract relevant information about the system. In this step we learn something new about the system we study and, in many ways, this the key and most creative step in doing simulation-based research. Addressing the the analysis of simulation results could be a subject of a long series of tutorials.
