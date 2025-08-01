package com.medbot.db;

import com.medbot.model.DiagnosisHistory;
import com.medbot.model.Patient;

import java.sql.*;

import java.util.List;
import java.util.ArrayList;

public class PatientRepository {
    private static final String URL = "jdbc:mysql://127.0.0.1:3306/teamproject";
    private static final String USER = "root";
    private static final String PASSWORD = "root";

    public static Patient findById(String patientId) { // String 타입으로 변경
        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD)) {
            PreparedStatement stmt = conn.prepareStatement("SELECT age, gender, conditions FROM patient WHERE id = ?");
            stmt.setString(1, patientId); // <-- setString()으로 변경
            ResultSet rs = stmt.executeQuery();

            if (rs.next()) {
                return new Patient(
                        patientId, // 생성자도 String 타입으로 변경
                        rs.getInt("age"),
                        rs.getString("gender"),
                        rs.getString("conditions")
                );
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }
    
    public static void save(Patient patient) throws SQLException {
        // try-with-resources로 Connection 자동 닫기
        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD)) {
            String sql = "INSERT INTO patient (id, age, gender, conditions) VALUES (?, ?, ?, ?)";
            
            PreparedStatement stmt = conn.prepareStatement(sql);
            
            stmt.setString(1, patient.id);
            stmt.setInt(2, patient.age);
            stmt.setString(3, patient.gender);
            stmt.setString(4, patient.conditions);
            
            int rowsAffected = stmt.executeUpdate(); // 쿼리 실행 및 변경된 행 수 반환

            if (rowsAffected > 0) {
                System.out.println("환자 정보가 성공적으로 저장되었습니다.");
            } else {
                System.out.println("환자 정보 저장에 실패했습니다.");
            }
        }
    }
    
    public static List<DiagnosisHistory> findDiagnosisHistoryByPatientId(String patientId) throws SQLException {
        List<DiagnosisHistory> historyList = new ArrayList<>();
        String sql = "SELECT symptoms, diagnosis_result, chat_date FROM diagnosis_history WHERE patient_id = ? ORDER BY chat_date ASC";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement stmt = conn.prepareStatement(sql)) {
            
            stmt.setString(1, patientId);
            ResultSet rs = stmt.executeQuery();
            
            while (rs.next()) {
                historyList.add(new DiagnosisHistory(
                    rs.getString("symptoms"),
                    rs.getString("diagnosis_result"),
                    rs.getTimestamp("chat_date")
                ));
            }
        }
        return historyList;
    }
}
