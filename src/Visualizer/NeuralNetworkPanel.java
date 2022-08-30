package Visualizer;

import javax.swing.*;
import java.awt.*;

public class NeuralNetworkViz extends JPanel {
    int width, height;
    NeuralNetworkViz(){
        width = 500;
        height = 500;
        this.setPreferredSize(new Dimension(width,height));
    }

    public void paint(Graphics g) {
        Graphics2D g2D = (Graphics2D) g;
        g2D.drawLine(0,0, width, height);
    }

}
