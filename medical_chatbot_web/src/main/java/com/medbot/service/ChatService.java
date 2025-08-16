package com.medbot.service;

import com.medbot.domain.Patient;
import com.medbot.repository.PatientRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.Optional;

@Service
public class ChatService {

    @Autowired
    private PatientRepository patientRepository;

    public String processChatMessage(String message, String patientId) throws IOException {
        Patient patient = null;

        if (patientId != null && !patientId.isEmpty()) {
            Optional<Patient> optPatient = patientRepository.findById(patientId);
            if (optPatient.isPresent()) {
                patient = optPatient.get();
            }
        }

        // 여기서 patient 정보 기반으로 메시지 처리하고 외부 AI 서버 호출하는 로직 작성
        // 예시: AI 서버에 message + patient 정보 전달 후 응답 받기
        String response = callExternalAIModel(message, patient);

        return response;
    }

    private String callExternalAIModel(String message, Patient patient) {
        // 실제 AI 서버 호출 구현 (예: RestTemplate, WebClient 등)
        // 예시로 간단히 patient 정보 포함하여 반환
        String info = (patient != null) ? 
            "Patient info - ID: " + patient.getId() + ", Age: " + patient.getAge() : "No patient info";

        return "AI Response for message: '" + message + "'. " + info;
    }
}
