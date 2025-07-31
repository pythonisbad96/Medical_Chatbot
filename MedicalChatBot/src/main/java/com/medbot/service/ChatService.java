package com.medbot.service;

import com.medbot.model.Patient;
import org.json.JSONObject;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Scanner;

public class ChatService {
    public String askModel(String userInput, Patient patient) {
        try {
            URL url = new URL("http://localhost:5000/chat");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");

            JSONObject json = new JSONObject();
            json.put("input", userInput);
            json.put("age", patient.getAge());
            json.put("gender", patient.getGender());
            json.put("history", patient.getHistory());

            try (OutputStream os = conn.getOutputStream()) {
                os.write(json.toString().getBytes());
            }

            Scanner sc = new Scanner(conn.getInputStream());
            return sc.hasNext() ? sc.nextLine() : "응답 오류";

        } catch (Exception e) {
            e.printStackTrace();
            return "서버 오류";
        }
    }
}
