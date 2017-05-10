../bin/smooth2.exe -m c -n 1 -o triangle4d_1.txt triangle4d.txt
../bin/smooth2.exe -m c -n 2 -o triangle4d_2.txt triangle4d.txt
../bin/smooth2.exe -m c -n 3 -o triangle4d_3.txt triangle4d.txt
../bin/smooth2.exe -m c -n 4 -o triangle4d_4.txt triangle4d.txt
python txt2gp2.py triangle4.txt triangle4.dat
python txt2gp2.py triangle4d.txt triangle4d.dat
python txt2gp2.py triangle4d_1.txt triangle4d_1.dat
python txt2gp2.py triangle4d_2.txt triangle4d_2.dat
python txt2gp2.py triangle4d_3.txt triangle4d_3.dat
python txt2gp2.py triangle4d_4.txt triangle4d_4.dat
rm -f triangle4d_1.txt triangle4d_2.txt triangle4d_3.txt triangle4d_4.txt 
gnuplot triangle4.gnu
rm -f triangle4.dat triangle4d.dat triangle4d_1.dat triangle4d_2.dat triangle4d_3.dat triangle4d_4.dat 
