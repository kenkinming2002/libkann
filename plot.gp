set datafile separator comma
set style data lines

set multiplot layout 1,3

set xlabel "Time"

set title "Creatures Count"
plot 'statistics.csv' using (column("Time")):(column("Creatures Count - Healthy")) title "Healthy", \
     'statistics.csv' using (column("Time")):(column("Creatures Count - All")) title "All"

set title "Age"
plot 'statistics.csv' using (column("Time")):(column("Age - LowerQuartile")) title "LowerQuartile", \
     'statistics.csv' using (column("Time")):(column("Age - Median")) title "Median", \
     'statistics.csv' using (column("Time")):(column("Age - UpperQuartile")) title "UpperQuartile"

set title "Mating Count"
plot 'statistics.csv' using (column("Time")):(column("Mating Count - LowerQuartile")) title "LowerQuartile", \
     'statistics.csv' using (column("Time")):(column("Mating Count - Median")) title "Median", \
     'statistics.csv' using (column("Time")):(column("Mating Count - UpperQuartile")) title "UpperQuartile"
