using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using _35_2_Fedorova_neuro.ModelNeuroNet;

namespace _35_2_Fedorova_neuro
{
    public partial class Form1 : Form
    {
        double[] InputData = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        
        NeuroNet net;

        public Form1()
        {
            InitializeComponent();
            net = new NeuroNet(NetworkMode.Demo);
        }

        void ChangeStatus(Button b, int i)
        {
            if (b.BackColor == Color.White)
            {
                b.BackColor = Color.RoyalBlue;
                InputData[i] = 1;
            }
            else
            {
                b.BackColor = Color.White;
                InputData[i] = 0;
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
            string strSample = numericUpDown_digit.Value.ToString();

            for (int i = 0; i < InputData.Length; i++)
            {
                strSample += " " + InputData[i].ToString();
            }
            strSample += '\n';

            File.AppendAllText(pathFileTrainSample, strSample);
        }

        private void button_rec_Click(object sender, EventArgs e)
        {
            net.ForwardPass(net, InputData);
            label1.Text = net.Fact.ToList().IndexOf(net.Fact.Max()).ToString();
        }
    }
}
