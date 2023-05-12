<h1> Mobility Models Applied to Mobile Data from Madrid 2022 </h1>

<p>
	The present analysis compares Visitation Law, Radiation, and Gravity models using mobile phone data from Spain. Only data from Madrid is used, where movement across 119 municipios (neighbourhoods/councils) over the period of a month (January 2022, 31 days) was recorded in the form of number of visits from an origin to a location. Data was averaged over the whole month for analysis and only travellers from home destination were kept (i.e people travelling from residence municpio to any other municipio).

<h1> Models </h1>
<h2>Visitation Model</h2>
<p> According to the visitation law of human mobility the number of unique locations visited by an individual is proportional to the square root of the total number of visits they make (Schläpfer et al., 2021). Thus the magnitude of a flow of visitors given a distance <i>r</i> and a frequency of visitation <i>f</i> can be approximated as: <i>&mu;</i> = <i>&rho;</i>(<i>rf</i>)<sup>2</sup> , where <i>r</i> = distance between origin and destination, <i>f</i> = once per day and the "spectral flow" <i>&rho;</i> =  <i>N/A</i>, where <i>N</i> are the visitors counts and <i>A</i> = <i>2&pi;r&delta;r</i>, the area of displacement where we assume (for present analysis) the incremental distance <i>&delta;r</i> = 1km. So, the average number of visitors per day can be estimated as: </p>

<p align="center"> <i>V</i> = <i>&mu;A</i>/<i>r<sup>2</sup>ln(f<sub>max</sub>/f<sub>min</sub>)</i></p>

Presently we use the entire matrix of areas <i>A</i>, slightly abusing notation <i>A</i> = <i>A<sub>ij</sub></i> . We use the entire matrix of magnitudes <i>&mu;</i> and radi <i>r</i> in the same manner. Where <i>f<sub>min</sub> = F/N/31</i>, and <i>f<sub>max</sub> = F/N</i>, where <i>F</i> is the average number of visits from origin to location over 1 month and <i>N</i> is the average number of visitors over the same month. That is, <i>f<sub>min</sub></i> indicates the average minimum frequency of visitation per person during a month and <i>f<sub>max</sub></i> the maximum.

<h2>Radiation Model</h2>
<p> The Radiation Model is parameter-free and estimates the number of visitors from one location (i.e. municipio) to another (Simini et al., 2013). The model assumes movement from origin to location as arising from a Binomial process, this with mean (i.e. estimate) <i>pT<sub>i</sub></i> and variance <i>pT<sub>i</sub>(1 - p)</i>, where <i>T<sub>i</sub></i> is the estimated total number of commuters from origin <i>i</i>, such as <i>T<sub>i</sub> = M<sub>i</sub>(N<sub>c</sub> / N)</i>, where <i>N<sub>c</sub></i> is the total number of visitors, <i>N</i> the total population of Madrid, and <i>M<sub>i</sub></i> are the populations at the origins (i.e. populations of municipios), equivalent to the populations at the destinations <i>N<sub>j</sub></i> (i.e. each municipio is origin to another and destination from another municipio). Then the model is defined as:</p>

<p align="center"> <i>E(T<sub>ij</sub>) = T<sub>i</sub>M<sub>i</sub>N<sub>j</sub> / (M<sub>i</sub> + S<sub>ij</sub>)(Mi<sub>i</sub> + N<sub>j</sub> + S<sub>ij</sub>)</i> </p>

<p>Where <i>S<sub>ij</sub></i> corresponds to the matrix containing the populations surrounding the origin within a radius <i>R<sub>ij</sub></i> which is equivalent to the distance between origin and destination.</p>

<h2>Gravity Model</h2>
Finally, we run a Gravity model over the vector of travellers from Mostoles (example municipio) to the each municipio (119 including Mostoles). We used a Normalised Power Law Gravity Model (e.g. see Simini et al, 2021). The model is sampled via a Negative Binomial distribution, as it may be better for over-dispersed data, especially as it has distinct parameters for mean and variance (more relevant to appropriately capture uncertainty).

 <p align="center"> &sigma;<sub>&theta;</sub>, &sigma;<sub>&omega;</sub>, &sigma;<sub>&gamma;</sub> ~ Exponential(3)</p>
 <p align="center"> &theta;<sub>m</sub> ~ HalfNormal(&sigma;<sub>&theta;</sub>)</p>
 <p align="center"> &omega;<sub>m</sub> ~ HalfNormal(&sigma;<sub>&omega;</sub>)</p>
 <p align="center"> &gamma;<sub>m</sub> ~ HalfNormal(&sigma;<sub>&gamma;</sub>)</p>
<p align="center"> &lambda; = &theta;<sub>m</sub>M<sub>i</sub>(N<sub>j</sub><sup>&omega;<sub>m</sub></sup>D<sup>-&gamma;<sub>m</sub></sup> /  &sum;N<sub>j</sub><sup>&omega;<sub>m</sub></sup>D<sup>-&gamma;<sub>m</sub></sup>)</p>
<p align="center"> &sigma; ~ Exponential(50) </p>
<p align="center"> y<sub>m</sub> ~ NegativeBinomial(&lambda;, &sigma;) </p>

Where sub-index m = [1...119] municpios, D = distance matrix from municpio to municipio, M<sub>i</sub> = population at origins, and N<sub>j</sub> = populations at destinations. The model sampled well with R^ ~ 1 and effective sample sizes over 500 for all parameters.

<h1> Results </h1>

<p> Results indicate that the Visitation Model outperforms the other models and the Gravity Models outperforms the Radiation Model according to an similarity index (SSI) metric (Schläpfer et al., 2021), defined as <i>SSI = 2&sum;<sub>ij</sub>min(E,O)/(&sum;<sub>ij</sub>E + &sum;<sub>ij</sub>O)</i>, where <i>E</i> is the estimated average number of visitors and <i>O</i> is the observed average number of visitors.</p>

<p align="center">
	<img src="compare_models.png" width="800" height="500" />
<p>

<p align="center">
	<img src="plot_madrid_municipio_observed.png" width="800" height="500" />
<p>

<p align="center">
	<img src="plot_madrid_municipio_visitation.png" width="800" height="500" />
<p>

<p align="center">
	<img src="plot_madrid_municipio_radiation.png" width="800" height="500" />
<p>

<p align="center">
	<img src="plot_madrid_municipio_gravity.png" width="800" height="500" />
<p>


<h1> Conclusion </h1>

<p> The Gravity and Visitation models perform much better than the Radiation model. However, the Gravity Model tends to underperform at large values, while the Visitation Model slightly underperforms at low values. More exploration of metrics and approximations may be required to better assess the models. The Gravity model may offer more leeway as its parametric form may be optimised by parameter adjustment, prior predictive checks and other techniques. Nevertheless, it is possible to explore parametric forms of the Visitation Model as well, which may outperform the other models in the long run.   </p>

<H1> References </H1>
<P>Schläpfer, M., Dong, L., O’Keeffe, K. et al. The universal visitation law of human mobility. Nature 593, 522–527 (2021). https://doi.org/10.1038/s41586-021-03480-9 </p>
<P>Simini, F., González, M., Maritan, A. et al. A universal model for mobility and migration patterns. Nature 484, 96–100 (2012). https://doi.org/10.1038/nature10856 </p>
<p>Simini, F., Barlacchi, G., Luca, M. et al. A Deep Gravity model for mobility flows generation. Nat Commun 12, 6576 (2021). https://doi.org/10.1038/s41467-021-26752-4</p>
