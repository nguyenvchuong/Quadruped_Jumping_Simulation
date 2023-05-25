clear all
load('backflipFull_A1_1ms_h0_d-60.mat')
load('backflipFull_A1_1ms_h0_d-60_cartesian.mat')
csvwrite('backflipFull_A1_1ms_h0_d-60.csv', data1)
csvwrite('backflipFull_A1_1ms_h0_d-60_cartesian.csv', data2)