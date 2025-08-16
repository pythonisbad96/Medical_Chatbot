package com.medbot.repository;

import com.medbot.domain.DiagnosisHistory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface DiagnosisHistoryRepository extends JpaRepository<DiagnosisHistory, Integer> {
    List<DiagnosisHistory> findTop5ByPatientIdOrderByChatDateDesc(String patientId);
    List<DiagnosisHistory> findByPatientIdOrderByChatDateDesc(String patientId);
    Page<DiagnosisHistory> findAllByPatientId(String patientId, Pageable pageable);
}


