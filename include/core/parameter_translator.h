#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <unordered_map>

template <typename T>
class ParameterTranslator
{
public:
  ParameterTranslator(const std::unordered_map<std::string, T> objects)
    : m_objects(objects)
  {}

  boost::optional<T>
  get_value(const std::string &value)
  {
    auto it = m_objects.find(value);
    if (it == m_objects.cend())
      {
        throw std::domain_error("Value not allowed: " + std::string(value));
      }
    return it->second;
  }

  boost::optional<std::string>
  put_value(const T &value)
  {
    for (auto it = m_objects.begin(); it != m_objects.end(); ++it)
      {
        if (it->second == value)
          {
            return it->first;
          }
      }
    throw std::domain_error("Value not found");
  }

private:
  std::unordered_map<std::string, T> m_objects;
};