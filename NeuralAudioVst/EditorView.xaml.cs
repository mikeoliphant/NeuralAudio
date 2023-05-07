using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Window;

namespace NeuralAudioVst
{
    /// <summary>
    /// Interaction logic for EditorView.xaml
    /// </summary>
    public partial class EditorView : UserControl
    {
        public EditorView()
        {
            InitializeComponent();

            DataContextChanged += EditorView_DataContextChanged;
        }

        private void EditorView_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if (!String.IsNullOrEmpty((DataContext as NeuralAudioPlugin).ModelPath))
            {
                LoadButton.Content = Path.GetFileName((DataContext as NeuralAudioPlugin).ModelPath);
            }
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new System.Windows.Forms.OpenFileDialog();
            dialog.DefaultExt = ".nam";
            dialog.Filter = "NAM Models|*.nam";
            dialog.ValidateNames = true;

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                try
                {
                    (DataContext as NeuralAudioPlugin).LoadModel(dialog.FileName);

                    LoadButton.Content = Path.GetFileName(dialog.FileName);
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message, "Error Loading Model");
                }
            }
        }
    }
}
