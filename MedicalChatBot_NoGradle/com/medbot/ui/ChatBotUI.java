package com.medbot.ui;

import com.medbot.model.Patient;
import com.medbot.db.PatientRepository;
import com.medbot.service.ChatService;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.sql.SQLException;

public class ChatBotUI extends JFrame {
	private JTextArea chatArea;
	private JTextField inputField;
	private JTextField patientIdField;
	private Patient currentPatient;

    // 회원가입 필드는 JoinDialog로 이동했으므로 삭제
	
	public void createAndShowGUI() {
		JFrame frame = new JFrame("의료 챗봇 MedBot");
		chatArea = new JTextArea(20, 50);
		inputField = new JTextField(40);
		patientIdField = new JTextField(5);

		chatArea.setEditable(false);
		
		// 상단 패널을 간소화합니다. 회원가입 버튼만 남깁니다.
		JPanel topPanel = new JPanel();
	    topPanel.add(new JLabel("환자 ID:"));
	    topPanel.add(patientIdField);
	    JButton loadButton = new JButton("불러오기");
	    topPanel.add(loadButton);
	    JButton insertButton = new JButton("회원가입");
	    topPanel.add(insertButton);

		// 하단: 채팅 입력
		JPanel bottomPanel = new JPanel();
		bottomPanel.add(inputField);
		JButton sendButton = new JButton("전송");
		bottomPanel.add(sendButton);

		frame.setLayout(new BorderLayout());
		frame.add(topPanel, BorderLayout.NORTH);
		frame.add(new JScrollPane(chatArea), BorderLayout.CENTER);
		frame.add(bottomPanel, BorderLayout.SOUTH);
		frame.pack();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);

		// 환자 정보 불러오기 (기존 코드와 동일)
		loadButton.addActionListener((ActionEvent e) -> {
			String patientIdText = patientIdField.getText();
			if (patientIdText.trim().isEmpty()) {
				chatArea.append("❌ 환자 ID를 입력해주세요.\n");
				return;
			}
			currentPatient = PatientRepository.findById(patientIdText);
			if (currentPatient != null) {
				chatArea.append("✅ 환자 정보: " + currentPatient + "\n");
			} else {
				chatArea.append("❌ 환자 정보를 찾을 수 없습니다.\n");
			}
		});

		// 환자 정보 저장하기 (JoinDialog를 띄우는 코드로 변경)
		insertButton.addActionListener((ActionEvent e) -> {
		    JoinDialog joinDialog = new JoinDialog(frame, this); // ChatBotUI 객체를 전달
		    joinDialog.setVisible(true);
		});
		
		// 메시지 전송 (기존 코드와 동일)
		sendButton.addActionListener((ActionEvent e) -> {
			if (currentPatient == null) {
				chatArea.append("❗ 환자 정보를 먼저 불러오세요.\n");
				return;
			}

			String userInput = inputField.getText();
			chatArea.append("🙋 사용자: " + userInput + "\n");
			try {
				String reply = ChatService.sendToServer(userInput, currentPatient);
				chatArea.append("🩺 MedBot: " + reply + "\n");
			} catch (Exception ex) {
				chatArea.append("❌ 서버 통신 오류\n");
				ex.printStackTrace();
			}

			inputField.setText("");
		});
	}
	
	// JoinDialog에서 호출할 메서드
	public void appendChatArea(String message) {
		chatArea.append(message);
	}
}