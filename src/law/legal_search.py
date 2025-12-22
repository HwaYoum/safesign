import os
import requests
import xml.etree.ElementTree as ET
import json 
from dotenv import load_dotenv

# 1. 환경 설정: API 키 로드
load_dotenv()
MOLEG_API_KEY = os.getenv("MOLEG_API_KEY") 

def search_law_id(law_name):
    """
    법령 이름으로 ID를 검색하고 법령명(real_name)과 ID를 반환합니다. (JSON 응답 파싱)
    사용 API: lawSearch (type=json)
    """
    url = f"http://www.law.go.kr/DRF/lawSearch.do?OC={MOLEG_API_KEY}&target=eflaw&nw=3&query={law_name}&type=json" 
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        data = response.json() 
        
        laws = data.get("LawSearch", {}).get("law", []) 
        target = None
        
        if laws:
            exact_match = next((law for law in laws if law.get("법령명한글") == law_name), None)
            
            if exact_match:
                target = exact_match
            else:
                laws.sort(key=lambda x: len(x.get("법령명한글", "")))
                target = laws[0]
                
        if target:
            raw_id = target.get("법령ID")
            real_name = target.get("법령명한글")
            return str(int(raw_id)) if raw_id and raw_id.isdigit() else raw_id, real_name
    except requests.exceptions.RequestException as e:
        print(f"⚠️ ID 검색 및 요청 실패 ({law_name}): {e}")
    except json.JSONDecodeError:
        print(f"⚠️ ID 검색 JSON 파싱 실패 ({law_name}).")
    except Exception as e:
        print(f"⚠️ ID 검색 중 일반 오류 ({law_name}): {e}")
    return None, None

def get_law_content_xml(law_id):
    """
    법령 본문 XML을 가져와 raw content (bytes)로 반환합니다.
    사용 API: lawService (type=XML 요청)
    """
    if not law_id: return None
    
    # 법령 본문은 XML 포맷으로 요청
    url = f"http://www.law.go.kr/DRF/lawService.do?OC={MOLEG_API_KEY}&target=eflaw&ID={law_id}&type=XML" 
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"⚠️ 본문 XML 다운로드 실패 (ID:{law_id}): {e}")
        return None
    except Exception as e:
        print(f"⚠️ 본문 XML 다운로드 중 일반 오류 (ID:{law_id}): {e}")
        return None

def parse_articles_from_xml(xml_content):
    if xml_content is None:
        return []
    
    parsed_articles = []
    try:
        root = ET.fromstring(xml_content) 
        
        for unit in root.findall(".//조문단위"):
            # 조문이 아닌 경우(예: 부칙 등) 제외
            is_article = unit.find("조문여부")
            if is_article is not None and is_article.text != "조문":
                continue
                
            # '조문내용' 태그에 이미 항/호 번호가 포함된 전체 텍스트가 들어있는 경우가 많음
            article_content = unit.find("조문내용")
            if article_content is not None and article_content.text:
                full_text = article_content.text.strip()
                
                # 항(Paragraph) 정보가 별도로 나뉘어 있는 경우를 위해 항들을 순회
                paragraphs = []
                for hang in unit.findall(".//항"):
                    hang_text = ""
                    # 항내용 가져오기
                    content_elem = hang.find("항내용")
                    if content_elem is not None and content_elem.text:
                        hang_text = content_elem.text.strip()
                    
                    # 호(Item) 정보가 있다면 추가
                    for ho in hang.findall(".//호"):
                        ho_content = ho.find("호내용")
                        if ho_content is not None and ho_content.text:
                            hang_text += f"\n  {ho_content.text.strip()}"
                    
                    if hang_text:
                        paragraphs.append(hang_text)
                
                # 만약 세부 항/호 정보가 추출되었다면 그것을 사용, 아니면 조문내용 통째로 사용
                if paragraphs:
                    result_text = full_text + "\n" + "\n".join(paragraphs)
                else:
                    result_text = full_text
                    
                parsed_articles.append(result_text)
                
    except Exception as e:
        print(f"⚠️ XML 파싱 오류: {e}")
        
    return parsed_articles