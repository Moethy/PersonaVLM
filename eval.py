import json
import os
import copy
import sys

from PersonaVLM import PersonaVLMAgent
from PersonaVLM import model


def check_result(correct_answer, response):
    if not response:
        return False
    if correct_answer[1].lower() == response.split(')')[0].lower()[-1]:
        return True
    return False


def evaluate(final_result):
    total_correct = 0
    total_count = 0

    for key, results in final_result.items():
        correct = sum(1 for r, a, _ in results if check_result(a, r))
        total = len(results)
        if 'alignment' in key:
            total_correct_align = correct
            total_count_align = total
        total_correct += correct
        total_count += total
        accuracy = correct / total if total > 0 else 0
        print(f"Evaluation for {key}: {correct}/{total}, Accuracy: {accuracy:.4f}")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"Overall Result: {total_correct}/{total_count}, Accuracy: {overall_acc:.4f}")
    print(f"Alignment Result: {total_correct_align}/{total_count_align}, Accuracy: {total_correct_align / total_count_align:.4f}")
    print(f"Overall - Alignment Results: {total_correct-total_correct_align}/{total_count-total_count_align}, Accuracy: {(total_correct-total_correct_align)/(total_count-total_count_align):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--save_dir", type=str, required=False,
                        default='./output/32k_Persona_MME_results',
                        help="Path to the dataset JSON file.")
    parser.add_argument("--model_path", type=str, required=False,
                        default='./checkpoints/rl',
                        help="")
    parser.add_argument("--bench_context", type=str, required=False,
                        default="32k",
                        help="")
    args = parser.parse_args()
    final_result = {}
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, "eval_persona_mme.json")

    if args.bench_context == "32k":
        bench_dirs = ['./data/Persona-MME/samples/50', './data/Persona-MME/samples/100']
    else:
        bench_dirs = ['./data/Persona-MME/samples/200', './data/Persona-MME/samples/500']
    
    benchmark_data = json.load(open("./data/Persona-MME/Persona-MME.json", 'r'))

    user_paths = []
    for dir in bench_dirs:
        user_paths += [os.path.join(dir, user_id) for user_id in sorted(os.listdir(dir)[:50])]
    print(f"Total number of users to evaluate: {len(user_paths)}")

    mllm = model.Qwenvl7B(model_path=args.model_path)
    for idx, user_path in enumerate(user_paths):
        print(idx, user_path)
        user_data = json.load(open(os.path.join(user_path, "user_data.json"), "r"))
        class UserConfig:
            USER_NAME = idx
            DATA_STORAGE_PATH = os.path.join(args.save_dir, f"eval_persona_mme/{USER_NAME}")
            PROFILE = user_data["profile"]
        user_config = UserConfig()
        print(user_config.DATA_STORAGE_PATH)
        model_for_evaluate = PersonaVLMAgent.PersonaVLM(user_config, model_path=args.model_path, force_retrieve=False, agent_model=mllm)

        img_idx = 0
        conv_total_num = 0
        for session_id, session in enumerate(user_data["sessions"]):
            if not session:
                continue
            for i in range(len(session)):
                conv_total_num += 1
                time = session[i]['time']
                user_msg = session[i]['user']
                bot_msg = session[i]['assistant']

                if "<img>" in user_msg:
                    img_path = os.path.join(user_path, os.path.basename(user_data["imgs"][img_idx]))
                    img_idx += 1
                    user_msg = user_msg.replace("<img>", "")
                else:
                    img_path = ""

                message = {
                    "query": user_msg,
                    "imgs": img_path,
                    "time": time
                }
                if sum([len(ss.get("interactions", [])) for ss in model_for_evaluate.data.get("sessions", [])]) >= conv_total_num:
                    continue
                else:
                    response, personality_info, reasoning_chain = model_for_evaluate.send_message(message, return_response_only=False, with_default_response=bot_msg)
        # get all the questions for evaluate
        questions = [q for q in benchmark_data if os.path.join(user_path, "user_data.json") == q['data_path']]
        print(f"Total number of questions: {len(questions)}")
        for question in questions:
            type = question['type']
            question_type = question['question_type']
            message = {
                "time": question["time"].split('(')[0].strip(),
                "query": question["query"],
            }
            info = question["choices"]
            best_answer = question["answer"]
            supplex = "\nOptions:\n" + '\n'.join([f"{key}. {value}" for key, value in info.items()]) + "\nPlease give your final answer (a), (b), (c) or (d)."
            message['query'] = message['query'] + supplex
            response, personality_info, reasoning_chain = model_for_evaluate.send_message(copy.deepcopy(message), return_response_only=False, update=False)
            print(message)
            print(f"Response: {response}")
            print(f"Answer: {best_answer}")
            final_result[f"{type}_{question_type}"] = final_result.get(f"{type}_{question_type}", [])
            final_result[f"{type}_{question_type}"].append([response, best_answer, reasoning_chain])
            
            if "alignment" in question:
                for i in range(2):
                    message = {
                                "time": question["time"].split('(')[0].strip(),
                                "query": question["query"],
                            }
                    best_answer = '(a)' if i == 0 else '(b)'
                    _tmp = question["alignment"]['chosen'] if i == 0 else question["alignment"]['rejected']
                    supplex = (
                        "Evaluate the following response based on the user's personality, which was revealed from the conversation history.\n"
                        f"\nResponse to Evaluate:\n\"{_tmp}\"\n"
                        "\nQuestion:\nDoes the provided response align with and adapt to the user's personality?\n"
                        "\nOptions:\n(a) Yes, it aligns well.\n(b) No, it misaligns.\n"
                        "\nPlease provide only the final letter of your choice: (a) or (b)."
                    )

                    message['query'] = message['query'] + supplex
                    response, personality_info, reasoning_chain = model_for_evaluate.send_message(copy.deepcopy(message), return_response_only=False, update=False)
                    print(message)
                    print(f"Response: {response}")
                    print(f"Answer: {best_answer}")
                    final_result[f"alignment"] = final_result.get(f"alignment", [])
                    final_result[f"alignment"].append([response, best_answer, reasoning_chain])
        evaluate(final_result)
        with open(save_path, 'w') as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        mllm.clear_conversation_history()
