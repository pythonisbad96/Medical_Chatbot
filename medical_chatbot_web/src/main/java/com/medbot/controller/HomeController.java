package com.medbot.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.ui.Model;

@Controller
public class HomeController {
	// ✅ application.properties에서 API URL을 주입받습니다.
    @Value("${app.api-url}")
    private String apiUrl;
	
    @GetMapping("/home")
    public String home(Model model) {
        // templates/home.html 파일을 찾아 렌더링
    	model.addAttribute("apiUrl", apiUrl);
        return "home";
    }
    
    @GetMapping("/chatbot")
    public String chatbot(Model model) {
    	// templates/chatbot.html 로 연결
    	model.addAttribute("apiUrl", apiUrl);
    	return "chatbot"; 
    }
}
