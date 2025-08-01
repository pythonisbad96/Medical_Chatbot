package com.medbot;

import com.medbot.ui.ChatBotUI;

public class Main {
    public static void main(String[] args) {
        javax.swing.SwingUtilities.invokeLater(() -> {
            new ChatBotUI().createAndShowGUI();
        });
    }
}
