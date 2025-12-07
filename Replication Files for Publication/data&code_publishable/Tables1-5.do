** Stata 15.1 and above
** 2023-09-11

clear all
set more off

global datadir "" 
// Please enter the path of the folder where the codes and the datasets are located.

cd "$datadir"

*==================================================================================================================================*

****************************************************** Summary Statistics **********************************************************

*==================================================================================================================================*


** table 1

use "subsample\firm information.dta",clear

sort year ambnumber
by year: gen numberoffirms=_N
by year: egen failurenextyear=sum(failure)
keep year numberoffirms failurenextyear
duplicates drop
insobs 1
replace year=2013 if year==.
tsset year
gen numberoffailureevents=L.failurenextyear
gen failurerate=numberoffailureevents/L.numberoffirms

insobs 1
egen total=sum(numberoffirms)
egen totalfailure=sum(numberoffailureevents)
gen totalfailurerate=totalfailure/total
tostring year,replace
replace year="total number of observations" if year=="."
replace numberoffirms=total if year=="total number of observations"
replace numberoffailureevents=totalfailure if year=="total number of observations"
replace failurerate=totalfailurerate if year=="total number of observations"
keep year numberoffirms numberoffailureevents failurerate

export delimited "tables and figures/table1.csv",replace


** table 2

use "subsample\firm information.dta",clear

sort country ambnumber
by country: gen numberofobservations=_N
by country: egen numberoffailureevents=sum(failure)
gen failurerate=numberoffailureevents/numberofobservations
keep country numberofobservations numberoffailureevents failurerate
duplicates drop

insobs 1
replace country="total" if country==""
egen total=sum(numberofobservations)
egen totalfailure=sum(numberoffailureevents)
gen totalfailurerate=totalfailure/total

replace numberofobservations=total if country=="total"
replace numberoffailureevents=totalfailure if country=="total"
replace failurerate=totalfailurerate if country=="total"
keep country numberofobservations numberoffailureevents failurerate

export delimited "tables and figures/table2.csv",replace


** table 3

use "subsample\firm information.dta",clear

estpost tabstat ///
profitorlossbeforetax profitorlossaftertax rop roa roe roebeforetax roabeforetax movingaverageofroa ///
movingaverageofrop movingaverageofroe movingaverageofroabeforetax movingaverageofroebeforetax ///
riskedadjustedroa riskedadjustedroe riskedadjustedroabeforetax riskedadjustedroebeforetax ///
grosspremiumswritten netpremiumswritten totalassets totalliabilities ///
totaldebtors grosstechnicalreserves nettechnicalreserves numberofemployees ///
capitalsurplus netpremiumswrittencapitalsurplus nettechnicalreservescapitalsurpl ///
grosspremiumswrittencapitalsurpl grosstechnicalreservescapitalsur totaldebtorscapitalsurplus ///
capitalsurplustotalassets leverageratio totaldebtorstotalassets ///
totalinvestments investmentasset totalinvestmenttotalliabilities ///
movingstandarddeviationofroa movingstandarddeviationofrop movingstandarddeviationofroe ///
movingstandarddeviationofroabefo movingstandarddeviationofroebefo ///
liquidassets liquidassetsnettechnicalreserves liquidassetstotalliabilities ///
mutual composite life affiliated changeinasset changeinnetpremium changeingrosspremium ///
premiumretainratio nettechnicalreservesgrosstechnic numberoflinesofbusiness numberofpremiumregions ///
, stat(mean min median max sd) col(stat)
esttab . using "tables and figures/table3.csv", replace noobs nonum nomti cells("mean(fmt(a2)) min(fmt(a2)) p50(fmt(a2)) max(fmt(a2)) sd(fmt(a2))") csv


** table 4

insheet using "macro.csv", clear comma
estpost tabstat ///
realgdpgrowth changeinyearlywage growthofindustryproduction growthofcapital ///
growthofimport growthofexport growthofconsumptionpercapita ///
growthofgovexpenditure growthoftaxrevenue unemploymentrate changeinunemploymentrate ///
domesticprivatecreditgdp domesticcreditgdp insurancedensity insurancepenetration ///
changeinpopulation laborforcepopulation growthofpopulationwithaccesstoin growthoflaborinagricultureindust ///
inflation yearlyreturnofmsci changeinimfoneyeartbillrate changeinexchangerate ///
gdp yearlywageinsurancesector gdppercapita changeincurrentaccount capitalratioofbanks growthofcarbonemission ///
, stat(mean min p50 max sd) col(stat)
esttab . using "tables and figures/table4.csv", replace noobs nonum nomti cells("mean(fmt(a2)) min(fmt(a2)) p50(fmt(a2)) max(fmt(a2)) sd(fmt(a2))") csv


** table 5

insheet using "yield.csv", clear comma
estpost tabstat y v5 v6 v7 v8 v9 v10 v11 v12 v13, stat(mean min p50 max sd) col(stat)
esttab . using "tables and figures/table5.csv", replace label noobs nonum nomti cells("mean(fmt(a2)) min(fmt(a2)) p50(fmt(a2)) max(fmt(a2)) sd(fmt(a2))") csv

