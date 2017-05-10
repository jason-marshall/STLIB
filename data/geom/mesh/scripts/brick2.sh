../bin/smooth3.exe -o brick2d_1.txt brick2d.txt
../bin/smooth3.exe -o brick2d_2.txt brick2d_1.txt
../bin/smooth3.exe -o brick2d_3.txt brick2d_2.txt
../bin/smooth3.exe -o brick2d_4.txt brick2d_3.txt
python txt2gp3.py brick2.txt brick2.dat
python txt2gp3.py brick2d.txt brick2d.dat
python txt2gp3.py brick2d_1.txt brick2d_1.dat
python txt2gp3.py brick2d_2.txt brick2d_2.dat
python txt2gp3.py brick2d_3.txt brick2d_3.dat
python txt2gp3.py brick2d_4.txt brick2d_4.dat
rm -f brick2d_1.txt brick2d_2.txt brick2d_3.txt brick2d_4.txt
gnuplot brick2.gnu
rm -f brick2.dat brick2d.dat brick2d_1.dat brick2d_2.dat brick2d_3.dat brick2d_4.dat
