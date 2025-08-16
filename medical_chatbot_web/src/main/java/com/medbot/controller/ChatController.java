package com.medbot.controller;

import com.medbot.domain.Patient;
import com.medbot.domain.DiagnosisHistory;
import com.medbot.repository.PatientRepository;
import com.medbot.repository.DiagnosisHistoryRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors; // ★ 반드시 import

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final PatientRepository patientRepository;
    private final DiagnosisHistoryRepository historyRepository;
    private final RestTemplate restTemplate = new RestTemplate();

    public ChatController(PatientRepository patientRepository,
                          DiagnosisHistoryRepository historyRepository) {
        this.patientRepository = patientRepository;
        this.historyRepository = historyRepository;
    }

    @Value("${python.api.url}")
    private String pythonApiUrl;

    @PostMapping
    public ResponseEntity<?> chat(
            @RequestParam String patientId,
            @RequestBody ChatRequest request
    ) {
        String symptoms = request.getMessage();

        // 1) 환자 정보
        Patient patient = patientRepository.findById(patientId)
                .orElseThrow(() -> new RuntimeException("환자 정보를 찾을 수 없습니다."));

        // 2) 최근 상담 5건
        List<DiagnosisHistory> recentHistory =
                historyRepository.findTop5ByPatientIdOrderByChatDateDesc(patientId);

        // 3) 프롬프트 구성
        String profileInfo = String.format(
                "[환자 프로필]\n- ID: %s\n- 나이: %d\n- 성별: %s\n- 기저질환: %s",
                patient.getId(),
                patient.getAge(),
                patient.getGender(),
                patient.getConditions() != null ? patient.getConditions() : "없음"
        );

        String historyText = recentHistory.isEmpty()
                ? "이전 상담 없음."
                : recentHistory.stream()
                .map(h -> String.format("[이전 상담]\n증상: %s\n예측 병명: %s",
                        h.getSymptoms(), h.getPredictedDiagnosis()))
                .collect(Collectors.joining("\n\n"));

        String finalPrompt = profileInfo + "\n\n" +
                historyText + "\n\n" +
                "[새 증상]\n" + symptoms + "\n\n" +
                "아래 형식으로 답변해주세요:\n" +
                "예측 병명:\n" +
                "병명 정의:\n" +
                "추천 진료과:\n" +
                "예방 및 관리:\n" +
                "추가 정보:\n";

        // 4) Python 호출
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        Map<String, String> body = new HashMap<>();
        body.put("prompt", finalPrompt);
        HttpEntity<Map<String, String>> entity = new HttpEntity<>(body, headers);

        ResponseEntity<PythonResponse> pyRes = restTemplate.exchange(
                pythonApiUrl + "/chat",
                HttpMethod.POST,
                entity,
                PythonResponse.class
        );
        if (pyRes.getBody() == null) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("AI 응답이 없습니다.");
        }
        PythonResponse ai = pyRes.getBody();

        // 5) DB 저장
        DiagnosisHistory history = new DiagnosisHistory();
        history.setPatientId(patient.getId());
        history.setSymptoms(symptoms);
        history.setPredictedDiagnosis(ai.getPredictedDiagnosis());
        history.setDiagnosisDefinition(ai.getDiagnosisDefinition());
        history.setRecommendedDepartment(ai.getRecommendedDepartment());
        history.setPreventionManagement(ai.getPreventionManagement());
        history.setAdditionalInfo(ai.getAdditionalInfo());
        historyRepository.save(history);

        return ResponseEntity.ok(ai);
    }

    // ===== 내부 DTO들 (수동 getter/setter) =====
    public static class ChatRequest {
        private String message;
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }

    public static class PythonResponse {
        private String predictedDiagnosis;
        private String diagnosisDefinition;
        private String recommendedDepartment;
        private String preventionManagement;
        private String additionalInfo;

        public String getPredictedDiagnosis() { return predictedDiagnosis; }
        public void setPredictedDiagnosis(String predictedDiagnosis) { this.predictedDiagnosis = predictedDiagnosis; }

        public String getDiagnosisDefinition() { return diagnosisDefinition; }
        public void setDiagnosisDefinition(String diagnosisDefinition) { this.diagnosisDefinition = diagnosisDefinition; }

        public String getRecommendedDepartment() { return recommendedDepartment; }
        public void setRecommendedDepartment(String recommendedDepartment) { this.recommendedDepartment = recommendedDepartment; }

        public String getPreventionManagement() { return preventionManagement; }
        public void setPreventionManagement(String preventionManagement) { this.preventionManagement = preventionManagement; }

        public String getAdditionalInfo() { return additionalInfo; }
        public void setAdditionalInfo(String additionalInfo) { this.additionalInfo = additionalInfo; }
    }
}
