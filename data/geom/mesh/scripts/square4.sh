../bin/smooth2.exe -b -o square4_1.txt square4.txt
../bin/smooth2.exe -b -n 5 -o square4_5.txt square4.txt
../bin/smooth2.exe -b -n 10 -o square4_10.txt square4.txt
../bin/smooth2.exe -b -n 100 -o square4_100.txt square4.txt
python txt2gp2.py square4.txt square4.dat
python txt2gp2.py square4_1.txt square4_1.dat
python txt2gp2.py square4_5.txt square4_5.dat
python txt2gp2.py square4_10.txt square4_10.dat
python txt2gp2.py square4_100.txt square4_100.dat
rm -f square4_1.txt square4_5.txt square4_10.txt square4_100.txt
gnuplot square4.gnu
rm -f square4.dat square4_1.dat square4_5.dat square4_10.dat square4_100.dat
