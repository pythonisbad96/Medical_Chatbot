package com.medbot.controller;

import com.medbot.domain.DiagnosisHistory;
import com.medbot.repository.DiagnosisHistoryRepository;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
public class DiagnosisHistoryController {

	private final DiagnosisHistoryRepository repo;

	public DiagnosisHistoryController(DiagnosisHistoryRepository repo) {
		this.repo = repo;
	}

	@PostMapping("/api/diagnosis-history")
	public ResponseEntity<?> save(@RequestBody Map<String, String> body, HttpSession session) {
		String loginId = (String) session.getAttribute("LOGIN_ID");
		if (loginId == null)
			return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("로그인이 필요합니다.");

		String patientId = body.get("patientId");
		if (patientId == null || !loginId.equals(patientId)) {
			return ResponseEntity.status(HttpStatus.FORBIDDEN).body("본인 계정의 기록만 저장할 수 있습니다.");
		}

		DiagnosisHistory dh = new DiagnosisHistory();
		dh.setPatientId(patientId);
		dh.setSymptoms(body.getOrDefault("symptoms", ""));
		dh.setPredictedDiagnosis(body.getOrDefault("predictedDiagnosis", ""));
		dh.setDiagnosisDefinition(body.getOrDefault("diagnosisDefinition", ""));
		dh.setRecommendedDepartment(body.getOrDefault("recommendedDepartment", ""));
		dh.setPreventionManagement(body.getOrDefault("preventionManagement", ""));
		dh.setAdditionalInfo(body.getOrDefault("additionalInfo", ""));
		// chatDate는 @PrePersist에서 자동 세팅

		DiagnosisHistory saved = repo.save(dh);
		return ResponseEntity.ok(saved);
	}
}
