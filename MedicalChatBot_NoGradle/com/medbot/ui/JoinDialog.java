package com.medbot.ui;

import com.medbot.model.Patient;
import com.medbot.db.PatientRepository;

import javax.swing.*;
import java.awt.*;
import java.sql.SQLException;

public class JoinDialog extends JDialog {
    private JTextField idField;
    private JTextField ageField;
    private JTextField conditionsField;
    private JRadioButton maleButton; // JRadioButton으로 변경
    private JRadioButton femaleButton; // JRadioButton으로 변경
    private ChatBotUI parentUI;

    public JoinDialog(JFrame parent, ChatBotUI parentUI) {
        super(parent, "회원가입", true);
        this.parentUI = parentUI;
        
        // 레이아웃 변경: 성별 버튼을 위한 공간 추가
        setLayout(new GridLayout(6, 2, 10, 10));
        
        idField = new JTextField(10);
        ageField = new JTextField(10);
        conditionsField = new JTextField(10);
        
        // JRadioButton 생성
        maleButton = new JRadioButton("남자");
        femaleButton = new JRadioButton("여자");

        // ButtonGroup을 사용하여 둘 중 하나만 선택되도록 설정
        ButtonGroup genderGroup = new ButtonGroup();
        genderGroup.add(maleButton);
        genderGroup.add(femaleButton);
        
        // 성별 버튼을 담을 패널
        JPanel genderPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        genderPanel.add(maleButton);
        genderPanel.add(femaleButton);

        JButton saveButton = new JButton("저장");
        JButton cancelButton = new JButton("취소");

        add(new JLabel("ID:"));
        add(idField);
        add(new JLabel("나이:"));
        add(ageField);
        add(new JLabel("성별:"));
        add(genderPanel); // 패널을 추가
        add(new JLabel("기저질환:"));
        add(conditionsField);
        add(saveButton);
        add(cancelButton);

        saveButton.addActionListener(e -> savePatient());
        cancelButton.addActionListener(e -> dispose());

        pack();
        setLocationRelativeTo(parent);
    }

    private void savePatient() {
        try {
            String id = idField.getText().trim();
            String ageText = ageField.getText().trim();
            String conditions = conditionsField.getText().trim();
            
            // 1. 필수 정보(ID, 나이, 성별)가 비어있는지 확인
            if (id.isEmpty() || ageText.isEmpty() || (!maleButton.isSelected() && !femaleButton.isSelected())) {
                JOptionPane.showMessageDialog(this, "ID, 나이, 성별은 필수 정보입니다.", "경고", JOptionPane.WARNING_MESSAGE);
                return;
            }

            // 2. ID가 영어만 포함하는지 검증
            if (!id.matches("[a-zA-Z]+")) {
                JOptionPane.showMessageDialog(this, "ID는 영어만 입력 가능합니다.", "경고", JOptionPane.WARNING_MESSAGE);
                return;
            }
            
            // 3. 기저질환이 공란일 경우 "없음"으로 설정
            if (conditions.isEmpty()) {
                conditions = "없음";
            }
            
            // 4. JRadioButton에서 선택된 값 가져오기
            String gender;
            if (maleButton.isSelected()) {
                gender = "m";
            } else { // femaleButton.isSelected()
                gender = "f";
            }
            
            int age = Integer.parseInt(ageText);
            
            Patient newPatient = new Patient(id, age, gender, conditions);
            PatientRepository.save(newPatient);

            parentUI.appendChatArea("✅ 환자 정보가 성공적으로 저장되었습니다. ID: " + id + "\n");
            dispose();

        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "나이는 숫자만 입력해주세요.", "경고", JOptionPane.WARNING_MESSAGE);
        } catch (SQLException ex) {
            JOptionPane.showMessageDialog(this, "데이터베이스 오류: " + ex.getMessage(), "오류", JOptionPane.ERROR_MESSAGE);
            ex.printStackTrace();
        }
    }
}