# SubgridClumping
 
A Code to derive the parameters for the global, in-homogeneous and stochastic clumping model and then compute the clumping factor for large low-resolution N-body simulations smoothed on a regular grid. Our code is meant for the <i>CUBEP3M</i> simulation. If you wish to use a different inputs, please contact the developper.
<br><br>
<img src="https://github.com/micbia/SubgridClumping/blob/main/results/AnClumpMic_190829_6.3Mpc_nc1200-so-n-MCPR_NEW2/noc8_bins5/plot/7.305_6.3_1.667_nc1200_clumping.png"> 
See publication about this work https://arxiv.org/abs/2101.01712.
<br><br>
Our framework is devided into two main codes.

&emsp;&#9654;&emsp;<b>AnalyseSubGridClumping.py</b>:<br>For a given small high-resolution simulation, it derives the three clumping model parameters. The variables to change are in the same file, they are the following:
<ul>
    <li><b>boxSize</b>: is the small box size in cMpc/h.</li>
    <li><b>redshift</b>: the list of redshift of the small box simulation.</li>
    <li><b>resLB</b>: the desired resolution (correspond to the large box resolution).</li>
    <li><b>noc</b>: number of coarsening (suggested to be > 8).</li>
    <li><b>MaxBin</b>: binning of the stochastic model (set to be 5).</li>
</ul> 

&emsp;&#9654;&emsp; <b>SimulateClumping.py</b>:<br>For the given density field of a large low-resolution simulation, it computes a clumping factor cube (same mesh-size as input) for the three models. The variables to change are in the same file, they are the following:

<ul>
    <li><b>boxSize_LB</b>: is the box size in cMpc/h.</li>
    <li><b>meshSize_LB</b>: size of the density regular grid.</li>
    <li><b>LB_path</b>: the directory of the smoothed density.</li>
    <li><b>output_path</b>: the directory, where to store the computed clumping cubes.</li>

</ul> 

<br><br>
Once the variables are changed the code can be run by simply:
<br>
&emsp;&#9654;&emsp; python AnalyseSubGridClumping.py
&emsp;&#9654;&emsp; python SimulateClumping.py
<br><br>
At the moment there are parameters available for a simulation with resolution 1.667 Mpc/h in:
<ul>
    <li>./results/AnClumpMic_190829_6.3Mpc_nc1200-so-n-MCPR_NEW2/noc8_bins5</li>
</ul>
so there is no need to run the first code.