package com.medbot.service;

import com.medbot.db.PatientRepository;
import com.medbot.model.DiagnosisHistory;
import com.medbot.model.Patient;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser; // JSONParser를 임포트해야 합니다.
import org.json.simple.parser.ParseException; // 예외 처리를 위해 임포트합니다.

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.sql.SQLException;
import java.util.List;

public class ChatService {
    public static String sendToServer(String message, Patient patient) throws IOException, SQLException {
    	// AI 모델에 전달할 전체 프롬프트 구성
        StringBuilder fullPrompt = new StringBuilder();

        // 1. 환자의 기본 정보 추가
        fullPrompt.append("나이: ").append(patient.age)
                  .append(", 성별: ").append(patient.gender)
                  .append(", 기저질환: ").append(patient.conditions)
                  .append("인 환자입니다. ");

        // 2. 과거 진단 기록을 조회하여 프롬프트에 추가
        List<DiagnosisHistory> history = PatientRepository.findDiagnosisHistoryByPatientId(patient.id);
        if (!history.isEmpty()) {
            fullPrompt.append("과거 진단 기록은 다음과 같습니다. ");
            for (DiagnosisHistory h : history) {
                fullPrompt.append("이전에 ").append(h.getSymptoms()).append(" 증상으로 ")
                          .append(h.getDiagnosisResult()).append(" 진단을 받았습니다. ");
            }
        }
        
        // 3. 현재 사용자의 메시지를 추가
        fullPrompt.append("이번에 챗봇에게 입력한 증상은 \"").append(message).append("\"입니다. ");
        fullPrompt.append("이 증상에 대한 병명을 알려주세요.");

        System.out.println("AI 모델에 전달할 최종 프롬프트: " + fullPrompt.toString()); // 디버깅용
    	
    	URL url = new URL("http://localhost:5000/chat");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);

        JSONObject json = new JSONObject();
        json.put("message", message);

        JSONObject patientJson = new JSONObject();
        patientJson.put("age", patient.age);
        patientJson.put("gender", patient.gender);
        patientJson.put("conditions", patient.conditions);
        json.put("patient", patientJson);

        try (OutputStream os = conn.getOutputStream()) {
            os.write(json.toString().getBytes());
        }

        BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        StringBuilder response = new StringBuilder();
        String line;
        while ((line = in.readLine()) != null)
            response.append(line);
        in.close();

        // 1. JSONParser를 사용해서 문자열을 파싱합니다.
        JSONParser parser = new JSONParser();
        try {
            JSONObject jsonResponse = (JSONObject) parser.parse(response.toString());

            // 2. get() 메소드를 사용하고 String으로 캐스팅합니다.
            return (String) jsonResponse.get("response");
        } catch (ParseException e) {
            // 파싱 오류 발생 시 예외 처리
            e.printStackTrace();
            return "오류: 서버 응답을 파싱할 수 없습니다.";
        }
    }
}
