** Stata 15.1 and above
** 2023-09-11

clear all
set more off

global datadir ""
// Please enter the path of the folder where the codes and the datasets are located.

cd "$datadir"

*==================================================================================================================================*

****************************************************** Subsample Generation ********************************************************

*==================================================================================================================================*

insheet using "firm information.csv",clear comma
save "subsample\firm information.dta",replace


** subsample: life
use "subsample\firm information.dta",clear
keep if internationalcompanytype=="Life"
drop if (benefitspaid==. | netexpenseslife==. )
gen selectcountry=country if failure==1
sort country selectcountry
by country: replace selectcountry=selectcountry[_N] if selectcountry=="" & selectcountry[_N]!=""
drop if selectcountry==""
drop selectcountry
save "subsample\firm information_life.dta",replace
export delimited "subsample\firm information_life.csv",replace


** subsample: nonlife
use "subsample\firm information.dta",clear
keep if internationalcompanytype=="Non-Life"
drop if (netexpenses==.  )
gen selectcountry=country if failure==1
sort country selectcountry
by country: replace selectcountry=selectcountry[_N] if selectcountry=="" & selectcountry[_N]!=""
drop if selectcountry==""
drop selectcountry
save "subsample\firm information_nonlife.dta",replace
export delimited "subsample\firm information_nonlife.csv",replace


** subsample: drop_mergers
use "subsample\firm information.dta",clear
drop if failure_event=="Merged"
gen selectcountry=country if failure==1
sort country selectcountry
by country: replace selectcountry=selectcountry[_N] if selectcountry=="" & selectcountry[_N]!=""
drop if selectcountry==""
drop selectcountry
save "subsample\firm information_dropmergers.dta",replace
export delimited "subsample\firm information_dropmergers.csv",replace

