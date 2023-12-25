using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.IO;
using _35_2_Fedorova_neuro.ModelNeuroNet;
using System.Windows.Forms.DataVisualization.Charting;

namespace _35_2_Fedorova_neuro
{
    public partial class Form1 : Form
    {
        private double[] inputData = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        private NeuroNet net;

        public double[] NetOutput { set => labelOutput.Text = "Цифра: " + value.ToList().IndexOf(value.Max()).ToString(); }
        Series s;

        public Form1()
        {
            InitializeComponent();
            net = new NeuroNet(NetworkMode.Demo);

            Series s = new Series("Error")
            {
                ChartType = SeriesChartType.Line,
                BorderWidth = 2
            };

            chart_errors.Series.Add(s);

        }

        void ChangeStatus(Button b, int i)
        {
            if (b.BackColor == Color.White)
            {
                b.BackColor = Color.RoyalBlue;
                inputData[i] = 1;
            }
            else
            {
                b.BackColor = Color.White;
                inputData[i] = 0;
            }
        }

        #region pixel button handlers
        private void button1_Click(object sender, EventArgs e)
        {
            ChangeStatus(button1, 0);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            ChangeStatus(button2, 1);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            ChangeStatus(button3, 2);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            ChangeStatus(button4, 3);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            ChangeStatus(button5, 4);
        }

        private void button6_Click(object sender, EventArgs e)
        {
            ChangeStatus(button6, 5);
        }

        private void button7_Click(object sender, EventArgs e)
        {
            ChangeStatus(button7, 6);
        }

        private void button8_Click(object sender, EventArgs e)
        {
            ChangeStatus(button8, 7);
        }

        private void button9_Click(object sender, EventArgs e)
        {
            ChangeStatus(button9, 8);
        }

        private void button10_Click(object sender, EventArgs e)
        {
            ChangeStatus(button10, 9);
        }

        private void button11_Click(object sender, EventArgs e)
        {
            ChangeStatus(button11, 10);
        }

        private void button12_Click(object sender, EventArgs e)
        {
            ChangeStatus(button12, 11);
        }

        private void button13_Click(object sender, EventArgs e)
        {
            ChangeStatus(button13, 12);
        }

        private void button14_Click(object sender, EventArgs e)
        {
            ChangeStatus(button14, 13);
        }

        private void button15_Click(object sender, EventArgs e)
        {
            ChangeStatus(button15, 14);
        }
        #endregion

        private void button_save_train_sample_Click(object sender, EventArgs e)
        {
            string pathFileTrainSample = AppDomain.CurrentDomain.BaseDirectory + "trainSample.txt";

            // желаемый отклик
            string strSample = numericUpDown_digit.Value.ToString();

            for (int i = 0; i < inputData.Length; i++)
            {
                strSample += " " + inputData[i].ToString();
            }
            strSample += '\n';

            File.AppendAllText(pathFileTrainSample, strSample);
        }

        private void button_rec_Click(object sender, EventArgs e)
        {
            net.ForwardPass(net, inputData);
            NetOutput = net.Fact;
        }

        private void button_train_Click(object sender, EventArgs e)
        {
            chart_errors.Series[1].Points.Clear();
            net.Train(net, (avgErrors) =>
            {
                for (int i = 0; i < avgErrors.Length; i++)
                {
                    chart_errors.Series[1].Points.AddXY(i, avgErrors[i]);
                }
            });
        }

        private void button17_Click(object sender, EventArgs e)
        {
            string pathFileTrainSample = AppDomain.CurrentDomain.BaseDirectory + "testSample.txt";

            // желаемый отклик
            string strSample = numericUpDown_digit.Value.ToString();

            for (int i = 0; i < inputData.Length; i++)
            {
                strSample += " " + inputData[i].ToString();
            }
            strSample += '\n';

            File.AppendAllText(pathFileTrainSample, strSample);
        }

        private void button16_Click(object sender, EventArgs e)
        {
            string[] testSample = File.ReadAllLines(AppDomain.CurrentDomain.BaseDirectory + "testSample.txt");

            string pathTestResult = AppDomain.CurrentDomain.BaseDirectory + "testResult.txt";

            string tmpStr = "";

            for (int i = 0; i < testSample.Length; i++)
            {
                string[] sample = testSample[i].Split(' ');

                int r = int.Parse(sample[0]);

                double[] tmpArr = new double[sample.Length - 1];
                for (int j = 1; j < sample.Length; j++)
                {
                    inputData[j - 1] = double.Parse(sample[j]);
                }

                net.ForwardPass(net, inputData);
    
                tmpStr += (r == net.Fact.ToList().IndexOf(net.Fact.Max())).ToString() + "\n";
            }


            File.WriteAllText(pathTestResult, tmpStr);
        }
    }
}
