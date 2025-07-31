package com.medbot.ui;

import com.medbot.service.ChatService;
import com.medbot.db.PatientRepository;
import com.medbot.model.Patient;

import javax.swing.*;
import java.awt.*;

public class ChatBotUI extends JFrame {
    private JTextArea chatArea;
    private JTextField inputField;
    private final ChatService chatService = new ChatService();

    public ChatBotUI() {
        setTitle("의료 챗봇 MedBot");
        setSize(500, 500);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        chatArea = new JTextArea();
        chatArea.setEditable(false);
        add(new JScrollPane(chatArea), BorderLayout.CENTER);

        inputField = new JTextField();
        inputField.addActionListener(e -> handleUserInput());
        add(inputField, BorderLayout.SOUTH);

        setVisible(true);
    }

    private void handleUserInput() {
        String input = inputField.getText();
        chatArea.append("🙋 사용자: " + input + "\n");

        Patient patient = PatientRepository.getPatientInfo("user1");
        String response = chatService.askModel(input, patient);

        chatArea.append("🩺 MedBot: " + response + "\n");
        inputField.setText("");
    }
}
